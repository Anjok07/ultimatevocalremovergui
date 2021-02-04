"""
Argumentparser version of UVR
"""
import argparse
import sys
from src.inference.converter_v4 import (VocalRemover, default_data)


parser = argparse.ArgumentParser(description='This is the terminal version of UVR.')
# -IO-
parser.add_argument('-i', '--inputs', metavar='InputPath/s', type=str, nargs='+',
                    required=True, dest='input_paths',
                    help='Path to music file/s')
parser.add_argument('-o', '--output', metavar='ExportPath', type=str, nargs=1,
                    required=True, dest='export_path',
                    help='Path to output directory')
# -Models-
parser.add_argument('-inst', '--instrumentalModel', metavar='InstrumentalPath', type=str, nargs=1,
                    required=True, dest='instrumentalModel',
                    help='Path to instrumental model')
parser.add_argument('-stacked', '--stackedModel', metavar='StackedPath', type=str, nargs=1,
                    required=False, dest='stackModel',
                    help='Path to stacked model')
# -Settings-
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
args = parser.parse_args()
# seperation_data = default_data.copy()
# seperation_data['input_paths'] = [r'']
# seperation_data['export_path'] = r''
# seperation_data['instrumentalModel'] = r''
# seperation_data['stackPasses'] = 0
# seperation_data['stackModel'] = r''

# vocal_remover = VocalRemover(data=seperation_data)
# vocal_remover.seperate_files()
