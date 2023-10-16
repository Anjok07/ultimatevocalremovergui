from datetime import datetime
import traceback

CUDA_MEMORY_ERROR = "CUDA out of memory"
CUDA_RUNTIME_ERROR = "CUDNN error executing cudnnSetTensorNdDescriptor"
DEMUCS_MODEL_MISSING_ERROR = "is neither a single pre-trained model or a bag of models."
ENSEMBLE_MISSING_MODEL_ERROR = "local variable \'enseExport\' referenced before assignment"
FFMPEG_MISSING_ERROR = """audioread\__init__.py", line 116, in audio_open"""
FILE_MISSING_ERROR = "FileNotFoundError"
MDX_MEMORY_ERROR = "onnxruntime::CudaCall CUDA failure 2: out of memory"
MDX_MODEL_MISSING = "[ONNXRuntimeError] : 3 : NO_SUCHFILE"
MDX_MODEL_SETTINGS_ERROR = "Got invalid dimensions for input"
MDX_RUNTIME_ERROR = "onnxruntime::BFCArena::AllocateRawInternal"
MODULE_ERROR = "ModuleNotFoundError"
WINDOW_SIZE_ERROR = "h1_shape[3] must be greater than h2_shape[3]"
SF_WRITE_ERROR = "sf.write"
SYSTEM_MEMORY_ERROR = "DefaultCPUAllocator: not enough memory"
MISSING_MODEL_ERROR = "'NoneType\' object has no attribute \'model_basename\'"
ARRAY_SIZE_ERROR = "ValueError: \"array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.\""
GPU_INCOMPATIBLE_ERROR = "no kernel image is available for execution on the device"
SELECT_CORRECT_GPU = "CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect."

CONTACT_DEV = 'If this error persists, please contact the developers with the error details.'

ERROR_MAPPER = {
    CUDA_MEMORY_ERROR:
                        ('The application was unable to allocate enough GPU memory to use this model. ' + 
                        'Please close any GPU intensive applications and try again.\n' + 
                        'If the error persists, your GPU might not be supported.') ,
    CUDA_RUNTIME_ERROR:
                        (f'Your PC cannot process this audio file with the segment size selected. Please lower the segment size and try again.\n\n{CONTACT_DEV}'),
    DEMUCS_MODEL_MISSING_ERROR:
                        ('The selected Demucs model is missing. ' + 
                        'Please download the model or make sure it is in the correct directory.'),
    ENSEMBLE_MISSING_MODEL_ERROR:
                        ('The application was unable to locate a model you selected for this ensemble.\n\n' + 
                        'Please do the following to use all compatible models:\n\n1. Navigate to the \"Updates\" tab in the Help Guide.\n2. Download and install the model expansion pack.\n3. Then try again.\n\n' + 
                        'If the error persists, please verify all models are present.'),
    FFMPEG_MISSING_ERROR:
                        ('The input file type is not supported or FFmpeg is missing. Please select a file type supported by FFmpeg and try again. ' + 
                        'If FFmpeg is missing or not installed, you will only be able to process \".wav\" files until it is available on this system. ' + 
                        f'See the \"More Info\" tab in the Help Guide.\n\n{CONTACT_DEV}'),
    FILE_MISSING_ERROR:
                        (f'Missing file error raised. Please address the error and try again.\n\n{CONTACT_DEV}'),
    MDX_MEMORY_ERROR:
                        ('The application was unable to allocate enough GPU memory to use this model.\n\n' + 
                        'Please do the following:\n\n1. Close any GPU intensive applications.\n2. Lower the set segment size.\n3. Then try again.\n\n' + 
                        'If the error persists, your GPU might not be supported.'),
    MDX_MODEL_MISSING:
                        ('The application could not detect this MDX-Net model on your system. ' + 
                        'Please make sure all the models are present in the correct directory.\n\n' + 
                        'If the error persists, please reinstall application or contact the developers.'),
    MDX_RUNTIME_ERROR:
                        ('The application was unable to allocate enough GPU memory to use this model.\n\n' + 
                        'Please do the following:\n\n1. Close any GPU intensive applications.\n2. Lower the set segment size.\n3. Then try again.\n\n' + 
                        'If the error persists, your GPU might not be supported.'),
    WINDOW_SIZE_ERROR:
                        ('Invalid window size.\n\n' + 
                        'The chosen window size is likely not compatible with this model. Please select a different size and try again.'),
    SF_WRITE_ERROR:
                        ('Could not write audio file.\n\n' + 
                        'This could be due to one of the following:\n\n1. Low storage on target device.\n2. The export directory no longer exists.\n3. A system permissions issue.'),
    SYSTEM_MEMORY_ERROR:
                        ('The application was unable to allocate enough system memory to use this model.\n\n' + 
                        'Please do the following:\n\n1. Restart this application.\n2. Ensure any CPU intensive applications are closed.\n3. Then try again.\n\n' + 
                        'Please Note: Intel Pentium and Intel Celeron processors do not work well with this application.\n\n' +
                        'If the error persists, the system may not have enough RAM, or your CPU might not be supported.'),
    MISSING_MODEL_ERROR:
                        ('Model Missing: The application was unable to locate the chosen model.\n\n' + 
                        'If the error persists, please verify any selected models are present.'),
    GPU_INCOMPATIBLE_ERROR:
                        ('This process is not compatible with your GPU.\n\n' + 
                        'Please uncheck \"GPU Conversion\" and try again'),
    SELECT_CORRECT_GPU:
                        ('Make sure you\'ve chosen the correct GPU.\n\n'
                        'Go to the "Settings Guide", click the "Additional Settings" tab and select the correct GPU device.'),
    ARRAY_SIZE_ERROR:
                        ('The application was not able to process the given audiofile. Please convert the audiofile to another format and try again.'),
}

def error_text(process_method, exception):
                 
    traceback_text = ''.join(traceback.format_tb(exception.__traceback__))
    message = f'{type(exception).__name__}: "{exception}"\nTraceback Error: "\n{traceback_text}"\n'
    error_message = f'\n\nRaw Error Details:\n\n{message}\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n'
    process = f'Last Error Received:\n\nProcess: {process_method}\n\n'

    for error_type, full_text in ERROR_MAPPER.items():
        if error_type in message:
            final_message = full_text
            break
    else:
        final_message = (CONTACT_DEV) 
        
    return f"{process}{final_message}{error_message}"

def error_dialouge(exception):
    
    error_name = f'{type(exception).__name__}'
    traceback_text = ''.join(traceback.format_tb(exception.__traceback__))
    message = f'{error_name}: "{exception}"\n{traceback_text}"'
    
    for error_type, full_text in ERROR_MAPPER.items():
        if error_type in message:
            final_message = full_text
            break
    else:
        final_message = (f'An Error Occurred: {error_name}\n\n{CONTACT_DEV}') 
    
    return final_message
