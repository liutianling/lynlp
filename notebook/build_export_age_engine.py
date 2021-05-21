from onnx import ModelProto
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)


def build_engine(onnx_path, shape=None):
    """
    This is the function to create the TensorRT engine
    Args:
        onnx_path : Path to onnx_file.
        shape : Shape of the input of the ONNX file.
    """
    if shape is None:
        shape = [1, 3, 224, 224]

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network,
                                                                                                  TRT_LOGGER) as parser:
        builder.max_workspace_size = (8 << 30)  # 4G
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())

        network.get_input(0).shape = shape
        engine = builder.build_cuda_engine(network)
        return engine


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


engine_name = 'fairface.plan'
onnx_path = "fairface.onnx"
batch_size = 1

model = ModelProto()
with open(onnx_path, "rb") as f:
    model.ParseFromString(f.read())

d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
shape = [batch_size, d0, d1, d2]
print("The default input shape is: ", shape)
engine = build_engine(onnx_path, shape=shape)
save_engine(engine, engine_name)

print("Finish.")
