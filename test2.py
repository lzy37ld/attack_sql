from accelerate import Accelerator
from accelerate.state import AcceleratorState


def main():
    accelerator = Accelerator()
    # accelerator.print(f"{AcceleratorState()}")
    accelerator.print(accelerator.state.deepspeed_plugin.gradient_clipping)


if __name__ == "__main__":
    main()