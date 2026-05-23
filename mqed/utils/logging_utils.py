# logging_bootstrap.py
from loguru import logger
import logging, sys
from pathlib import Path

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

def setup_loggers_hydra_aware():
    # Try to grab the Hydra runtime dir; fallback to CWD if not initialized (e.g., during unit tests)
    try:
        from hydra.core.hydra_config import HydraConfig
        if HydraConfig.initialized():
            outdir = Path(HydraConfig.get().runtime.output_dir)
            job_name = HydraConfig.get().job.name
        else:
            outdir = Path.cwd()
            job_name = "run"
    except Exception:
        outdir = Path.cwd()
        job_name = "run"

    outdir.mkdir(parents=True, exist_ok=True)
    logfile = outdir / f"{job_name}.log"

    logger.remove()
    logger.add(sys.stdout,
               level="INFO",
               format="<green>{time}</green> | <level>{message}</level>")
    logger.add(logfile, level="DEBUG", encoding="utf-8", backtrace=False, diagnose=False)

    # Bridge stdlib logging into Loguru so libraries show up too
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG)

    logger.success(f"Logging -> {logfile}")
    log_citation()
    return logfile


def log_citation():
    logger.info(
        "MQED-QD | If you use this software, please cite: "
        "Liu, G., Wang, S. and Chen, H.T., 2026. MQED-QD: An Open-Source Package for Quantum Dynamics Simulation in Complex Dielectric Environments. Journal of Chemical Theory and Computation."
    )
