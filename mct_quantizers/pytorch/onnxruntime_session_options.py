import onnxruntime as ort
from onnxruntime_extensions import get_library_path


def get_ort_session_options():
    opt = ort.SessionOptions()
    opt.register_custom_ops_library(get_library_path())
    return opt
