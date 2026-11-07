"""Run pip-audit with native OS certificate verification on this network."""

import runpy

import truststore


def main():
    # Use Windows' trusted roots instead of bypassing an enterprise CA chain.
    truststore.inject_into_ssl()
    runpy.run_module("pip_audit", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
