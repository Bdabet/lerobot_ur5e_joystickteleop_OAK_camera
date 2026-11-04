# openpi_client (vendored)

Minimal copy of `packages/openpi-client/src/openpi_client` from openpi
(https://github.com/THI-Robotics-Lab/openpi, Apache-2.0, see `LICENSE`), used by
`scripts/core/run_policy.py` to talk to the pi0.5 policy server.

Vendored instead of `pip install openpi-client` because that package pins `numpy<2`, which
would downgrade this env's numpy 2.2.6 and break LeRobot. The client works with numpy 2.

Only the files the robot client needs are included (websocket client, action chunk broker,
msgpack numpy codec, image tools). Extra dependencies: `msgpack`, `websockets`, `dm-tree`
(imported as `tree`), `pillow`, `typing_extensions`.
