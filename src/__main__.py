import sys
from src.object.parse import Config


if __name__ == "__main__":
    rutas = Config.parse_arguments(sys.argv[1:])
    print(rutas.input_path)
    print(rutas.output_path)
