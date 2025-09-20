import os
import pandas as pd
import re

class SummaryExtractor:
    def __init__(self, dir_path, astra_output_path, astra_output_file_name, summary_file_name, num_data):
        self.dir_path = dir_path
        self.astra_output_path = astra_output_path
        self.num_data = num_data
        self.astra_output_file_name = astra_output_file_name 
        self.summary_file_name = summary_file_name
        self.summary_file = None

    def open_summary_file(self):
        self.summary_file = open(self.dir_path + self.summary_file_name, 'w', encoding='UTF-8')

    def read_summary_or_error(self,file_name):
        num_lines = 8
        with open(file_name, 'rb') as file:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            buffer_size = 1024
            buffer = bytearray()

            for pos in range(file_size, 0, -buffer_size):
                if pos < buffer_size:
                    buffer_size = pos
                file.seek(pos - buffer_size, os.SEEK_SET)
                buffer[0:0] = file.read(buffer_size)

                if b'-SUMMARY' in buffer or b'-ERROR' in buffer or b'err' in buffer or b'-UNHAPPY' in buffer:
                    break

            lines = buffer.decode('utf-8', errors='ignore').splitlines()

            for line in reversed(lines):
                if '-SUMMARY' in line:
                    summary_index = lines.index(line)
                    return [line.strip() for line in lines[summary_index+7:summary_index + num_lines]]

                elif '-ERROR' in line or 'err' in line or 'UNHAPPY' in line:
                    print("# OUTPUT FILE IS ABNORMAL ")
                    self.summary_file.write("# OUTPUT FILE IS ABNORMAL \n")
                    return []

    def summary_file_writer(self):
        self.open_summary_file()
        for i in range(self.num_data):
            file_name = f"{self.astra_output_path}{self.astra_output_file_name}_{i}.out"
            print(f"{self.astra_output_file_name}_{i}")
            self.summary_file.write(file_name + '\n')
            summary_lines = self.read_summary_or_error(file_name)
            for line in summary_lines:
                self.summary_file.write(line + "\n")
                print(line)

        self.summary_file.close()

    
    def deleted_index_recoder(self):
        astra_summary = pd.read_csv(self.summary_file_name, header = None)

        # 오류 발생 ASTRA output index 기록
        deleted_rows = set()
        deleted_index = []

        for index, value in enumerate(astra_summary.values):
            if "# OUTPUT FILE IS ABNORMAL" in value[0]:
                # output file is abnormal 부분 바로 이전이 gd_cy1_.....숫자 이고 [index-1]은 이전줄을 의미, match.group(1)은 index 숫자를 의미

                match = re.search(f'{self.astra_output_file_name}_(\d+)\.out', astra_summary.iloc[index-1].values[0])
                if match:
                    print("Extracted number:", match.group(1))
                deleted_rows.add(index)
                deleted_index.append(match.group(1))
                if index > 0:
                    deleted_rows.add(index - 1)

        astra_summary.drop(index=deleted_rows, inplace=True)
        
        return deleted_index, astra_summary