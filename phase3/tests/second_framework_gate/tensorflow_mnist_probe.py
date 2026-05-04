#!/usr/bin/env python3
"""Milestone 05 TensorFlow GPU registration and bounded training probe."""

from __future__ import annotations

import json
import math
import time
import traceback
from typing import Any, Optional


def _json_safe_float(x: float) -> Optional[float]:
    if not math.isfinite(x):
        return None
    return x


def main() -> int:
    report: dict[str, Any] = {
        "overall_pass": False,
        "import_tensorflow": False,
        "tensorflow_version": None,
        "build_info": None,
        "physical_gpus": [],
        "logical_gpus": [],
        "memory_growth_enabled": False,
        "dataset": None,
        "used_gpu_for_training": False,
        "train_elapsed_sec": None,
        "history": None,
        "error": None,
        "traceback": None,
    }

    try:
        import tensorflow as tf

        report["import_tensorflow"] = True
        report["tensorflow_version"] = tf.__version__
        report["build_info"] = tf.sysconfig.get_build_info()
        physical = tf.config.list_physical_devices("GPU")
        for gpu in physical:
            tf.config.experimental.set_memory_growth(gpu, True)
        report["memory_growth_enabled"] = bool(physical)
        logical = tf.config.list_logical_devices("GPU")
        report["physical_gpus"] = [str(item) for item in physical]
        report["logical_gpus"] = [str(item) for item in logical]

        if not logical:
            raise RuntimeError("TensorFlow did not register a logical GPU")

        try:
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            report["dataset"] = "mnist"
            x = x_train[:1024].astype("float32") / 255.0
            y = y_train[:1024]
        except Exception as exc:  # pragma: no cover - field fallback
            report["dataset"] = f"synthetic_mnist_shape_fallback: {type(exc).__name__}: {exc}"
            x = tf.random.uniform([1024, 28, 28], dtype=tf.float32).numpy()
            y = tf.random.uniform([1024], minval=0, maxval=10, dtype=tf.int32).numpy()

        start = time.time()
        with tf.device("/GPU:0"):
            x_tf = tf.reshape(tf.convert_to_tensor(x[:4, :4, :4], dtype=tf.float32), [4, 16])
            y_tf = tf.convert_to_tensor(y[:4], dtype=tf.int32)
            y_one_hot = tf.one_hot(y_tf, depth=10, dtype=tf.float32)
            weights = tf.Variable(tf.zeros([10, 16], dtype=tf.float32))
            bias = tf.Variable(tf.zeros([10], dtype=tf.float32))

            # Avoid cuBLAS/cuDNN and large XLA reductions so the gate measures
            # TensorFlow GPU kernel execution through the current vGPU surface.
            logits = tf.reduce_sum(tf.expand_dims(x_tf, 1) * tf.expand_dims(weights, 0), axis=2) + bias
            probabilities = tf.nn.softmax(logits)
            loss = -tf.reduce_mean(tf.reduce_sum(y_one_hot * tf.math.log(probabilities + 1e-6), axis=1))
            error = probabilities - y_one_hot
            grad_weights = tf.reduce_mean(tf.expand_dims(error, 2) * tf.expand_dims(x_tf, 1), axis=0)
            grad_bias = tf.reduce_mean(error, axis=0)
            weights.assign_sub(0.01 * grad_weights)
            bias.assign_sub(0.01 * grad_bias)

            predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_tf), tf.float32))
            loss_value = float(loss.numpy())
            accuracy_value = float(accuracy.numpy())

        report["train_elapsed_sec"] = round(time.time() - start, 3)
        report["history"] = {
            "loss": [_json_safe_float(loss_value)],
            "accuracy": [_json_safe_float(accuracy_value)],
        }
        report["used_gpu_for_training"] = True
        report["overall_pass"] = True
    except Exception as exc:  # pragma: no cover - field diagnostics
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc(limit=8)

    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=str))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
