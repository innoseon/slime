#!/bin/bash

set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/root/slime}"

NNODES="${PET_NNODES:-${NNODES:-16}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
GPUS_PER_NODE="${PET_NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-6379}}"
DASHBOARD_PORT="${DASHBOARD_PORT:-$((MASTER_PORT + 1))}"

TEST_GPUS_PER_NODE="${TEST_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
TEST_TIMEOUT_SEC="${TEST_TIMEOUT_SEC:-600}"
TEST_TENSOR_MB="${TEST_TENSOR_MB:-64}"
TEST_ITERS="${TEST_ITERS:-5}"
TEST_BASE_PORT="${TEST_BASE_PORT:-$((MASTER_PORT + 100))}"
USE_EXTERNAL_RAY="${USE_EXTERNAL_RAY:-0}"
STOP_RAY_AFTER_TEST="${STOP_RAY_AFTER_TEST:-0}"
RESET_RAY_BEFORE_TEST="${RESET_RAY_BEFORE_TEST:-0}"

export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}}"
export MASTER_ADDR MASTER_PORT

cd "${SLIME_DIR}"

if [ "${RESET_RAY_BEFORE_TEST}" = "1" ]; then
    ray stop --force || true
fi

TEST_PY="/tmp/slime_ray_network_nccl_test.py"
cat > "${TEST_PY}" <<'PY'
import argparse
import json
import os
import socket
import subprocess
import threading
import time
from datetime import timedelta

import ray


def _node_ip():
    try:
        import ray._private.services

        return ray._private.services.get_node_ip_address()
    except Exception:
        return socket.gethostbyname(socket.gethostname())


@ray.remote(num_cpus=1)
class NodeProbe:
    def __init__(self):
        self.host = socket.gethostname()
        self.ip = _node_ip()
        self._server = None
        self._server_thread = None
        self._server_seen = []
        self._server_errors = []

    def info(self):
        try:
            nvidia_smi = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT, timeout=20)
        except Exception as exc:
            nvidia_smi = repr(exc)
        return {
            "host": self.host,
            "ip": self.ip,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": nvidia_smi.strip().splitlines(),
        }

    def start_tcp_server(self, expected_connections, timeout_sec):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", 0))
        self._server.listen(expected_connections)
        self._server.settimeout(timeout_sec)
        port = self._server.getsockname()[1]
        self._server_seen = []
        self._server_errors = []

        def _serve():
            deadline = time.time() + timeout_sec
            while len(self._server_seen) < expected_connections and time.time() < deadline:
                try:
                    conn, addr = self._server.accept()
                    with conn:
                        payload = conn.recv(256)
                        self._server_seen.append({"from": addr[0], "payload": payload.decode("utf-8", "replace")})
                        conn.sendall(b"ok")
                except socket.timeout:
                    break
                except Exception as exc:
                    self._server_errors.append(repr(exc))
                    break
            try:
                self._server.close()
            except Exception:
                pass

        self._server_thread = threading.Thread(target=_serve, daemon=True)
        self._server_thread.start()
        return {"host": self.host, "ip": self.ip, "port": port}

    def ping_peers(self, peers, timeout_sec):
        results = []
        for peer in peers:
            if peer["ip"] == self.ip and peer["host"] == self.host:
                continue
            started = time.time()
            try:
                with socket.create_connection((peer["ip"], peer["port"]), timeout=timeout_sec) as sock:
                    sock.sendall(f"{self.host}/{self.ip}".encode())
                    reply = sock.recv(16)
                results.append(
                    {
                        "peer": peer,
                        "ok": reply == b"ok",
                        "latency_ms": round((time.time() - started) * 1000, 2),
                    }
                )
            except Exception as exc:
                results.append({"peer": peer, "ok": False, "error": repr(exc)})
        return {"host": self.host, "ip": self.ip, "results": results}

    def tcp_server_results(self):
        if self._server_thread is not None:
            self._server_thread.join(timeout=1)
        return {"host": self.host, "ip": self.ip, "seen": self._server_seen, "errors": self._server_errors}


@ray.remote(num_cpus=1)
class CpuDistWorker:
    def run(self, rank, world_size, master_addr, master_port, timeout_sec):
        import torch
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        started = time.time()
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_sec),
        )
        tensor = torch.ones(1)
        dist.all_reduce(tensor)
        dist.barrier()
        elapsed = time.time() - started
        dist.destroy_process_group()
        return {
            "rank": rank,
            "host": socket.gethostname(),
            "ip": _node_ip(),
            "backend": "gloo",
            "value": float(tensor.item()),
            "elapsed_sec": round(elapsed, 3),
        }


@ray.remote(num_cpus=1, num_gpus=1)
class GpuDistWorker:
    def run(self, rank, world_size, master_addr, master_port, timeout_sec, tensor_mb, iters):
        import torch
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available on rank {rank}, host={socket.gethostname()}")
        torch.cuda.set_device(0)

        started = time.time()
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_sec),
        )
        init_elapsed = time.time() - started

        numel = max(1, tensor_mb * 1024 * 1024 // 4)
        times = []
        for _ in range(iters):
            tensor = torch.ones(numel, device="cuda", dtype=torch.float32)
            torch.cuda.synchronize()
            tic = time.time()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            times.append(time.time() - tic)
            expected = float(world_size)
            got = float(tensor[0].item())
            if got != expected:
                raise RuntimeError(f"rank {rank}: all_reduce got {got}, expected {expected}")
            dist.barrier()

        dist.destroy_process_group()
        return {
            "rank": rank,
            "host": socket.gethostname(),
            "ip": _node_ip(),
            "backend": "nccl",
            "cuda_visible_devices": cuda_visible,
            "init_elapsed_sec": round(init_elapsed, 3),
            "avg_all_reduce_sec": round(sum(times) / len(times), 4),
            "max_all_reduce_sec": round(max(times), 4),
            "tensor_mb": tensor_mb,
            "iters": iters,
        }


def _alive_nodes(expected_nnodes):
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    nodes = sorted(nodes, key=lambda n: n.get("NodeManagerAddress", ""))
    print(f"[ray] alive nodes: {len(nodes)}/{expected_nnodes}", flush=True)
    for n in nodes:
        print(
            json.dumps(
                {
                    "node_id": n.get("NodeID"),
                    "ip": n.get("NodeManagerAddress"),
                    "resources": n.get("Resources"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if len(nodes) < expected_nnodes:
        raise RuntimeError(f"Only {len(nodes)} Ray nodes are alive, expected {expected_nnodes}")
    return nodes


def _head_first(nodes, preferred_master_addr):
    for i, node in enumerate(nodes):
        if node.get("NodeManagerAddress") == preferred_master_addr:
            return [node] + nodes[:i] + nodes[i + 1 :]
    print(
        f"[warn] MASTER_ADDR={preferred_master_addr} is not a Ray NodeManagerAddress; "
        f"using {nodes[0].get('NodeManagerAddress')} as torch.distributed rank0 host.",
        flush=True,
    )
    return nodes


def _make_node_actor(cls, node):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return cls.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
    ).remote()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-nnodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--tensor-mb", type=int, default=64)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    ray.init(address="auto", logging_level="ERROR")
    nodes = _head_first(_alive_nodes(args.expected_nnodes), args.master_addr)
    master_addr = nodes[0]["NodeManagerAddress"]

    print("\n[node info]", flush=True)
    node_actors = [_make_node_actor(NodeProbe, node) for node in nodes[: args.expected_nnodes]]
    infos = ray.get([actor.info.remote() for actor in node_actors])
    for info in infos:
        print(json.dumps(info, ensure_ascii=False, sort_keys=True), flush=True)

    print("\n[tcp all-to-all between nodes]", flush=True)
    peers = ray.get(
        [
            actor.start_tcp_server.remote(expected_connections=args.expected_nnodes - 1, timeout_sec=args.timeout_sec)
            for actor in node_actors
        ]
    )
    pings = ray.get([actor.ping_peers.remote(peers, timeout_sec=10) for actor in node_actors])
    server_results = ray.get([actor.tcp_server_results.remote() for actor in node_actors])
    failed = [
        {"from": item["ip"], "to": r["peer"], "error": r.get("error")}
        for item in pings
        for r in item["results"]
        if not r["ok"]
    ]
    print(f"tcp pairs tested: {args.expected_nnodes * (args.expected_nnodes - 1)}, failed: {len(failed)}", flush=True)
    if failed:
        print(json.dumps(failed[:50], ensure_ascii=False, indent=2), flush=True)
        raise RuntimeError("TCP node-to-node connectivity test failed")
    print("tcp all-to-all ok", flush=True)
    for result in server_results:
        if result["errors"]:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)

    print("\n[gloo one rank per node]", flush=True)
    gloo_workers = [_make_node_actor(CpuDistWorker, node) for node in nodes[: args.expected_nnodes]]
    refs = [
        worker.run.remote(rank, args.expected_nnodes, master_addr, args.base_port, args.timeout_sec)
        for rank, worker in enumerate(gloo_workers)
    ]
    gloo_results = sorted(ray.get(refs), key=lambda x: x["rank"])
    for item in gloo_results:
        print(json.dumps(item, sort_keys=True), flush=True)
    if any(item["value"] != float(args.expected_nnodes) for item in gloo_results):
        raise RuntimeError("Gloo all_reduce returned an unexpected value")
    print("gloo all_reduce ok", flush=True)

    print("\n[nccl one rank per GPU]", flush=True)
    gpu_nodes = nodes[: args.expected_nnodes]
    gpu_workers = []
    for node in gpu_nodes:
        available = int(node.get("Resources", {}).get("GPU", 0))
        count = min(args.gpus_per_node, available)
        if count < args.gpus_per_node:
            raise RuntimeError(
                f"Node {node.get('NodeManagerAddress')} has {available} Ray GPUs, expected {args.gpus_per_node}"
            )
        for _ in range(count):
            gpu_workers.append(_make_node_actor(GpuDistWorker, node))

    world_size = len(gpu_workers)
    print(f"nccl world_size={world_size}, tensor_mb={args.tensor_mb}, iters={args.iters}", flush=True)
    refs = [
        worker.run.remote(rank, world_size, master_addr, args.base_port + 1, args.timeout_sec, args.tensor_mb, args.iters)
        for rank, worker in enumerate(gpu_workers)
    ]
    nccl_results = sorted(ray.get(refs), key=lambda x: x["rank"])
    for item in nccl_results:
        print(json.dumps(item, sort_keys=True), flush=True)
    print("nccl all_reduce ok", flush=True)
    print("\nALL TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
PY

RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

env = {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_DEBUG": os.environ["NCCL_DEBUG"],
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"],
    "NCCL_ASYNC_ERROR_HANDLING": os.environ["NCCL_ASYNC_ERROR_HANDLING"],
    "no_proxy": os.environ["no_proxy"],
    "MASTER_ADDR": os.environ["MASTER_ADDR"],
}
for name in (
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_IB_DISABLE",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TC",
    "NCCL_IB_SL",
):
    value = os.environ.get(name)
    if value:
        env[name] = value
print(json.dumps({"env_vars": env}))
PY
)"

submit_test() {
    ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python3 "${TEST_PY}" \
        --expected-nnodes "${NNODES}" \
        --gpus-per-node "${TEST_GPUS_PER_NODE}" \
        --master-addr "${MASTER_ADDR}" \
        --base-port "${TEST_BASE_PORT}" \
        --timeout-sec "${TEST_TIMEOUT_SEC}" \
        --tensor-mb "${TEST_TENSOR_MB}" \
        --iters "${TEST_ITERS}"
}

if [ "${USE_EXTERNAL_RAY}" = "1" ]; then
    submit_test
elif [ "${NODE_RANK}" = "0" ]; then
    ray start --head --node-ip-address "${MASTER_ADDR}" --port "${MASTER_PORT}" \
        --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats \
        --dashboard-host=0.0.0.0 --dashboard-port="${DASHBOARD_PORT}"

    python3 - <<PY
import sys
import time

import ray

expected = int("${NNODES}")
deadline = time.time() + int("${TEST_TIMEOUT_SEC}")
ray.init(address="auto", logging_level="ERROR")
while time.time() < deadline:
    alive = sum(1 for node in ray.nodes() if node.get("Alive"))
    print(f"Ray nodes alive: {alive}/{expected}", flush=True)
    if alive >= expected:
        sys.exit(0)
    time.sleep(5)
print(f"Timed out waiting for Ray nodes: {alive}/{expected}", flush=True)
sys.exit(1)
PY

    submit_test

    if [ "${STOP_RAY_AFTER_TEST}" = "1" ]; then
        ray stop --force
    fi
else
    sleep 5
    ray start --address="${MASTER_ADDR}:${MASTER_PORT}" \
        --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats --block
fi
