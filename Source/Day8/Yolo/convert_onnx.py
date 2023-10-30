from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="", help="convert yolo to onnx")

if __name__ == "__main__":
    model = YOLO("best.pt")
    model.export(format="onnx")
