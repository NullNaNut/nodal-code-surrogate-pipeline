import os
import pandas as pd
import numpy as np
from collections import Counter

class ResultFileParser():
    """
    Parses result files for a given nodal calculation code.
    
    Args:
        code_name (str): The name of the calculation code (e.g., 'ASTRA', 'MASTER', 'KARMA').
        result_file_name (str): The file name (not the full path) of the result file.
    """
    def __init__(self, code_name = None, result_file_name = None):
        if code_name is None:
            raise ValueError(
                "Error: You must specify the code name (e.g., 'ASTRA', 'MASTER', or 'KARMA')."
            )
        if result_file_name is None:
            raise ValueError(
                "Error: Result file name has not been provided."
            )

        self.code_name = code_name.upper() # Ensure code name is uppercase
        if self.code_name not in ['ASTRA', 'MASTER', 'KARMA']:
            raise ValueError(
                f"Error: Unsupported code name '{self.code_name}'. Supported codes are 'ASTRA', 'MASTER', and 'KARMA'."
            )
        self.result_file_name = result_file_name # Full path is not required, only the file name is needed.

# 나중에 추가할 예정
# class MasterResultParser(ResultFileParser):
# class KarmaResultParser(ResultFileParser):

class AstraResultParser(ResultFileParser):
    """Parses ASTRA result files.

    Args:
        ResultFileParser (class): Base class for result file parsing.
    """
    def __init__(self, code_name, result_file_name):
        super().__init__(code_name, result_file_name)
        self.lpd_matrix_authentic_batchID = None
        self.all_authentic_batch_ids = set()
        self.num_rows_in_quarter_lp = 5 # Number of rows in the assembly-wise quadrant loading pattern
        self.num_axial_planes = 24 # Number of axial planes in the assembly-wise distribution
        self.bottom_plane_index = 2 # Bottom plane index in the ASTRA output file
        self.axial_plane_indices = [x for x in range(2, 26)][::-1] # reversed, used in the "get_assemblywise_3d_distribution" method

        print(f"{self.__class__.__name__} instance has been successfully initialized.")
        print(f"  - ASTRA Output File: \'{self.result_file_name}\'")
        print(f"  - Number of rows in the assembly-wise quadrant loading pattern: {self.num_rows_in_quarter_lp}")

    def get_loading_pattern_and_batch_ids(self):
        """
        Parses the quadrant loading pattern (LP) matrix and authentic batch IDs
        from the %LPD_BCH and %LPD_B&C sections of the result file.

        The extracted authentic batch ID matrix and batch ID set are also stored
        as the member variables `self.lpd_matrix_authentic_batchID` and `self.all_authentic_batch_ids`.

        Returns:
            tuple:
                lpd_matrix_authentic_batchID (list[list[str]]): 2D matrix of authentic batch IDs.
                all_authentic_batch_ids (set[str]): Set of all unique authentic batch IDs found.
                
        Example:
            Example snippet from an input file:
            
            %LPD_BCH
            QROT(1)= A01 A03 A02 A02 A01
            QROT(2)= A03 A03 A04 A04 A02
            QROT(3)= A02 A04 A04 A02 A01
            QROT(4)= A02 A04 A05 A05
            QROT(5)= A01 A02 A05
            
            %LPD_B&C
            FUEL_DB(A01)  =  ENGD   A07   A
            FUEL_DB(A02)  =  ENGD   A08   A
            FUEL_DB(A03)  =  ENGD   A14   A
            FUEL_DB(A04)  =  ENGD   A82   A
            FUEL_DB(A05)  =  ENGD   A90   A
            AXCOMP(A01)   =  1*2   7*2  8*3  7*4   1*5 #
            AXCOMP(A02)   =  1*2   7*2  8*3  7*4   1*5 #
            AXCOMP(A03)   =  1*2   7*2  8*3  7*4   1*5 #
            AXCOMP(A04)   =  1*2   7*2  8*3  7*4   1*5 #
            AXCOMP(A05)   =  1*2   7*2  8*3  7*4   1*5 #
            
        Raises:
            FileNotFoundError: If the result file does not exist.
            UnicodeDecodeError: If the file cannot be read with the specified encoding.
        """
        with open(self.result_file_name, 'r', encoding='cp949') as result_file_object:
            lpd_matrix_pseudo_batchID = []
            batch_id_map = {}

            while True:
                line = result_file_object.readline()
                if not line:
                    break

                stripped_line = line.strip()
                
                if "%LPD_BCH" in stripped_line:
                    lpd_bch_flag = True
                    while lpd_bch_flag:
                        sub_line = result_file_object.readline()
                        if not sub_line:
                            break
                        row_number_qrot = int(sub_line.split('=')[0][-2]) 
                        temp_list_1d_line = sub_line.split()[1:]
                        if len(temp_list_1d_line) < self.num_rows_in_quarter_lp:
                            temp_list_1d_line += ['0'] * (self.num_rows_in_quarter_lp - len(temp_list_1d_line))
                        lpd_matrix_pseudo_batchID.append(temp_list_1d_line)
                        if row_number_qrot == self.num_rows_in_quarter_lp:
                            lpd_bch_flag = False
                            break

                if '%LPD_B&C' in stripped_line:
                    lpd_bnc_flag = True
                    while lpd_bnc_flag:
                        sub_line = result_file_object.readline()
                        if not sub_line:
                            break
                        if sub_line.strip().split('(')[:-1][0] == 'AXCOMP':
                            lpd_bnc_flag = False
                            break
                        temp_list_1d_line = sub_line.split()
                        current_draft_batch_id = temp_list_1d_line[0].split('(')[-1][:-1]

                        original_batch_id = None
                        for item in temp_list_1d_line[1:]:
                            if item.startswith("A") and item[1:].isdigit() and len(item) > 1:
                                original_batch_id = item
                                break

                        if original_batch_id is not None:
                            self.all_authentic_batch_ids.add(original_batch_id)
                            batch_id_map[current_draft_batch_id] = original_batch_id

        self.lpd_matrix_authentic_batchID = [
            [batch_id_map.get(draft_id, draft_id) for draft_id in row]
            for row in lpd_matrix_pseudo_batchID
        ]

        print("Function 'get_loading_pattern_and_batch_ids' executed successfully: loading patterns and batch IDs extracted.")
        
        return self.lpd_matrix_authentic_batchID, self.all_authentic_batch_ids
        
        
    def get_assembly_3d_distribution(self, assemblywise_result_type = None, step_index = None):
        """
        Function that constructs 3D matrix from section in the output card
    
        Args:
            file_name (str)
            section_acronym (str): e.g. '-P3D', '-B3D', '-XE453D'
            section_seq (int): sequential number per burnup step

        Return:
            3d quadrant assembly-wise distribution matrix (e.g., (24, 5, 5) for one step)
            Returns an empty list or raises an error on failure.
            
        Example:
            Example snippet from an input file:
            
            -P3D:1      3D ASSEMBLY-WISE POWER AND MAXIMUM PIN POWER DISTRIBUTIONS                0.0 MWD/MTU
                        JOB ID: -------, CODE VERSION: ASTRA 1.4.4, USER ID: limch, TIME: 11/29/2023 15:02:09                      

                        
                        FIRST  LINE: ASSEMBLY BATCH ID
                        SECOND LINE: ASSEMBLY POWER
                        THIRD  LINE: MAXIMUM PIN POWER
                        FOURTH LINE: PIN LOCATION (I,J)
                        
                        
                        
                        PLANE NUMBER:  25
                        
                        Y/X     E       F       G       H       J
                        
                        5       A05     A03     A04     A02     A01
                                0.4111  0.3854  0.3017  0.6573  0.5501
                                0.4439  0.4268  0.3827  0.7356  0.6907
                                5 10    5 10    17  1   13 10   4 11
                        
                        6       A03     A03     A05     A04     A02
                                0.3854  0.2650  0.5684  0.6927  0.5317
                                0.4268  0.3576  0.7008  0.7595  0.6797
                                8  5    17 17   14 13   8 13    4  6
                        
                        7       A04     A05     A04     A02     A01
                                0.3017  0.5684  0.7002  0.6788  0.4317
                                0.3827  0.7008  0.7756  0.7681  0.6210
                                17 17   13 14   10 13   4  6    3  5
                        
                        8       A02     A04     A02     A01
                                0.6573  0.6927  0.6788  0.4926
                                0.7356  0.7595  0.7681  0.6790
                                8 13    13  8   6  4    4  5
                        
                        9       A01     A02     A01
                                0.5501  0.5317  0.4317
                                0.6907  0.6797  0.6210
                                7  4    6  4    5  3

        Raises:
            ValueError: If the assemblywise_result_type is not specified or if step_index is invalid.
            FileNotFoundError: If the result file does not exist.
            UnicodeDecodeError: If the file cannot be read with the specified encoding.
            
        """
        if not os.path.isfile(self.result_file_name):
            raise FileNotFoundError(f"Error: The result file '{self.result_file_name}' does not exist.")
        
        if assemblywise_result_type is None:
            raise ValueError("Error: Please specify the assembly-wise result type to extract (e.g., '-P3D', '-B3D', '-XE453D').")
        
        if step_index is None or step_index < 1:
            raise ValueError("Error: 'step_index' must be specified and greater than 0.")

        single_3d_matrix = [] # 3D matrix to be returned
        found_section_start = False

        step_index_str = str(step_index)
        search_token = f"{assemblywise_result_type}:{step_index_str}" # e.g., '-P3D:1'

        try:
            encodings_to_try = ['CP949', 'utf-8', 'latin-1']
            file_lines = None
            for enc in encodings_to_try:
                try:
                    with open(self.result_file_name, 'r', encoding=enc) as f:
                        file_lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    print(f"Encoding {enc} failed, trying next...")
                    continue
                except FileNotFoundError:
                    print(f"Error: File not found at {self.result_file_name}")
                    return []
            if file_lines is None:
                print("Error: Could not read file with any attempted encoding.")
                return []

            line_iterator = iter(file_lines)

            # 1. Find section start
            while True:
                try:
                    current_line = next(line_iterator)
                    if search_token in current_line:
                        found_section_start = True
                        break
                except StopIteration:
                    print(f"Error: Reached end of file without finding section start for {assemblywise_result_type}:{step_index}")
                    return []

            if not found_section_start:
                print(f"Error: Section {assemblywise_result_type}:{step_index} not found.")
                return []

            # 2. Data extraction after section found
            planes_extracted_count = 0

            while planes_extracted_count < self.num_axial_planes: # 24 axial planes expected
                try:
                    sub_line = next(line_iterator)
                    split_line = sub_line.strip().split()

                    # Detect next section header (line starts with '-' and not the search_token)
                    if len(split_line) > 0 and split_line[0].startswith('-') and (search_token not in sub_line):
                        if not (split_line[0] == 'PLANE' or split_line[0].isdigit()):
                            print(f"Found start of a different section or irrelevant line: '{sub_line.strip()}'")
                            break

                    # PLANE line found
                    if len(split_line) >= 2 and split_line[0] == 'PLANE':
                        try:
                            plane_num = int(split_line[-1]) # e.g., PLANE 25
                            if plane_num in self.axial_plane_indices:
                                current_plane_matrix2d = []

                                # Skip 3 garbage lines
                                try:
                                    for _ in range(3):
                                        next(line_iterator)
                                except StopIteration:
                                    print("Warning: EOF while skipping garbage lines after PLANE header.")
                                    break

                                rows_in_plane = 0
                                valid_plane_data = True
                                while rows_in_plane < self.num_rows_in_quarter_lp: # 5 rows per plane (default)
                                    try:
                                        # Skip assembly ID line
                                        next(line_iterator)
                                        # Read data line
                                        assy_power_line = next(line_iterator)
                                        temp_list_str = assy_power_line.strip().split()

                                        # Data line should start with a number (possibly negative/decimal)
                                        if not temp_list_str or not temp_list_str[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                                            print(f"Warning: Expected data line but found '{assy_power_line.strip()}' at PLANE {plane_num}, Row {rows_in_plane+1}. Skipping plane.")
                                            valid_plane_data = False
                                            break

                                        temp_list_1d_line = list(map(float, temp_list_str))
                                        while len(temp_list_1d_line) < 5:
                                            temp_list_1d_line.append(0.0)
                                        current_plane_matrix2d.append(temp_list_1d_line[:5])
                                        rows_in_plane += 1

                                        # Skip bottom garbage lines
                                        if assemblywise_result_type in ["-P3D", "-B3D"]: num_garbage = 3
                                        elif assemblywise_result_type == "-XE453D": num_garbage = 1
                                        else: num_garbage = 0
                                        for _ in range(num_garbage): next(line_iterator)

                                    except StopIteration:
                                        print(f"Warning: EOF while processing data for PLANE {plane_num}, Row {rows_in_plane+1}.")
                                        valid_plane_data = False
                                        break
                                    except ValueError:
                                        print(f"Error converting data to float at PLANE {plane_num}, Row {rows_in_plane+1}: '{assy_power_line.strip()}'")
                                        valid_plane_data = False
                                        break

                                if valid_plane_data and len(current_plane_matrix2d) == self.num_rows_in_quarter_lp: # 5 rows expected per plane
                                    single_3d_matrix.append(current_plane_matrix2d)
                                    planes_extracted_count += 1
                                    if plane_num == self.bottom_plane_index: # If the bottom plane is reached, stop extraction
                                        break
                                elif not valid_plane_data:
                                    break

                        except ValueError:
                            pass
                        except IndexError:
                            pass

                except StopIteration:
                    print(f"Reached end of file unexpectedly during data extraction. Extracted {planes_extracted_count} planes.")
                    break

            # Result check and return
            if planes_extracted_count == self.num_axial_planes: # 24 planes expected
                # print(f"Successfully extracted {planes_extracted_count} planes for {section_acronym}:{section_seq}.")
                return np.array(single_3d_matrix)
            else:
                print(f"Error/Warning: Expected {self.num_axial_planes} planes, but extracted {planes_extracted_count} for {assemblywise_result_type}:{step_index}.")
                return []

        except Exception as e:
            print(f"An unexpected error occurred in assy_dist3D_extractor: {e}")
            import traceback
            traceback.print_exc()
            return []

 
 
    def get_summary_core_parameters(self): # Extracts data for all steps.
        """
        Parses the summary section at the end of the ASTRA result file and extracts core parameters for **all calculation steps**.

        The method reads the file from the end, searching for the '-SUMMARY' section (or error indicators).
        It then extracts specified core parameters (e.g., STEP, BURNUP, POWER, K-EFF, FQ, etc.) for each step and
        returns them as a list of lists (2D array), where each row corresponds to one calculation step.

        If an error string is found (e.g., '-ERROR', 'err', '-UNHAPPY'), the function returns error_flag=1 and None.

        Returns:
            tuple:
                error_flag (int): 0 if summary section is found and parsed successfully, 1 if an error section is found.
                collected_parameter_rows_filtered (list[list[float]] or None):
                    2D list containing selected parameters for each calculation step,
                    or None if an error was found in the file.

        Notes:
            - All steps present in the summary are parsed; select a specific step using its index from the returned list.
            - Parameters extracted include (by default): STEP, BURNUP, POWER, CBC, K-EFF, REACT, ASI, FR, FZ, FXY, FQ,
            TIN, TMOD, TFUEL, R4, R3, R2, R1 (see parameter_indices for column order).
            
        Example:
            Example snippet from an input file:
            
            -SUMMARY           OVERALL RESULT                                                   
                    JOB ID: a8r5rl5s, CODE VERSION: ASTRA 1.4.4, USER ID: limch, TIME: 11/29/2023 15:02:09                      


                       STEP  BURNUP  POWER   CBC    K-EFF     REACT   ASI  SADDLE   FR     FZ     FXY     FQ    TIN  TMOD   TFUEL XE SM ROD-POSITION
                             (MWD/)   (%)   (PPM)             (PCM)         INDEX                               (C)   (C)    (C)        S     R4    R3    R2    R1
                    --------------------------------------------------------------------------------------------------------------------------------------------------
                          1      0.  100.0    0.0 1.000000       0   0.194  0.000  1.441  1.474  1.485   2.096 295.5 310.4  561.2 EQ TR 100.0  34.2  84.2 100.0 100.0
                          2     50.  100.0    0.0 0.999999       0   0.195  0.000  1.439  1.479  1.484   2.097 295.5 310.4  546.0 EQ TR 100.0  35.4  85.4 100.0 100.0
                          3    500.  100.0    0.0 0.999999       0   0.188  0.000  1.427  1.476  1.474   2.074 295.5 310.2  518.8 EQ TR 100.0  45.2  95.2 100.0 100.0
                          4   1000.  100.0    0.0 0.999999       0   0.207  0.000  1.427  1.505  1.471   2.112 295.5 310.4  513.7 EQ TR 100.0  42.2  92.2 100.0 100.0
                          5   2000.  100.0    0.0 1.000001       0   0.214  0.000  1.425  1.510  1.465   2.119 295.5 310.5  506.8 EQ TR 100.0  36.3  86.3 100.0 100.0
                          6   3000.  100.0    0.0 1.000002       0   0.206  0.000  1.420  1.492  1.458   2.088 295.5 310.5  501.0 EQ TR 100.0  34.9  84.9 100.0 100.0
        
        Raises:
            FileNotFoundError: If the result file does not exist.
            UnicodeDecodeError: If the file cannot be read with the specified encoding.
        
        """
        if not os.path.isfile(self.result_file_name):
            raise FileNotFoundError(f"Error: The result file '{self.result_file_name}' does not exist.")
        
        with open(self.result_file_name, 'rb') as result_file_object:
            result_file_object.seek(0, os.SEEK_END)  # 파일의 끝으로 이동
            file_size = result_file_object.tell()
            buffer_size = 1024
            buffer = bytearray()

            # Read the file in reverse, starting from the end
            for pos in range(file_size, 0, -buffer_size):
                if pos < buffer_size:
                    buffer_size = pos
                result_file_object.seek(pos - buffer_size, os.SEEK_SET)
                buffer[0:0] = result_file_object.read(buffer_size)  # prepend the new data to the buffer

                # Find "-SUMMARY" or error strings
                if b'-SUMMARY' in buffer or b'-ERROR' in buffer or b'err' in buffer or b'-UNHAPPY' in buffer:
                    break

            # Convert the contents to a string
            lines = buffer.decode('utf-8', errors='ignore').splitlines()

            # Check for "-SUMMARY" or error strings
                # Extract all rows, so just extract the parameters for the required step using the index
            for line in reversed(lines):
                if '-SUMMARY' in line:
                    summary_index = lines.index(line) # '-SUMMARY' 문구 등장하는 라인
                    error_flag = 0 # 0: Success, 1: Error

                    core_parameter_rows = [line.strip().split() for line in lines[summary_index+7:summary_index + len(lines)]]
                    # 0: 'STEP', 1: 'BURNUP', 2: 'POWER', 3:'CBC', 4:'K-EFF', 5:'REACT', 6: 'ASI'
                    # 8: 'FR', 9: 'FZ', 10: 'FXY', 11: 'FQ'
                    # 12: 'TIN', 13: 'TMOD', 14: 'TFUEL',
                    # 18: 'R4', 19: 'R3', 20: 'R2', 21: 'R1'  
                    #
                    parameter_indices = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21]
                    
                    # shape: (step, len(parameter_indices))
                    core_parameter_rows = [
                        [float(row[i]) for i in parameter_indices] 
                        for row in core_parameter_rows 
                        if row and row[0] not in ('-HAPPY', 'JOB')  # line_list_split으로 수집한 line들은 parameter 뿐만 아닌, -HAPPY 또는 JOB 등의 불필요한 문구 존재
                    ]
                    
                    return error_flag, core_parameter_rows 
                
                elif '-ERROR' in line or 'err' in line or 'UNHAPPY' in line:
                    print("# OUTPUT FILE IS ABNORMAL.")
                    error_flag = 1 # 1: Error, 0: Success
                    
                    return error_flag, None

    
    def parameter_rows_to_dataframe(self, core_parameter_rows):
        """
        Converts a list of core parameter rows (as returned by get_summary_core_parameters)
        into a pandas DataFrame with appropriate column names.

        Args:
            core_parameter_rows (list[list[float]]): 2D list containing core parameter values for each calculation step.
        Returns:
            pandas.DataFrame: DataFrame with one row per calculation step and columns for each core parameter.
        """    
        column_names = ['STEP', 'BURNUP', 'POWER','CBC', 'K-EFF',
                    'REACT', 'ASI', 'FR', 'FZ', 'FXY',
                    'FQ', 'TIN', 'TMOD', 'TFUEL', 'R4',
                    'R3', 'R2', 'R1']

        df_parameter_rows = pd.DataFrame(core_parameter_rows, columns=column_names)

        return df_parameter_rows
    

# class DBParser(): 다른 코드들의 DB 구조 및 용법과 공통점이 별로 없으므로 상속은 생략

# 이제 뭘해야하나,
    # 1) result file에서 데이터 추출하는 클래스는 완성했고
    # 2) DB에서 각종 정보 처리하는 클래스
    # 3) 추출한 장전모형을 XS LP로 변환하는 클래스 필요

class AstraDBParser():
    def __init__(self, 
                 reactor_design_spec_file_name = "DB/iSMR_69FA_540MW_240CM_ENGD_LP_CN_r1",
                 cross_section_library_file_name = "DB/iSMR_GD_EN28f_r1.XS",
                 form_function_library_file_name = "DB/iSMR_GD_EN28f_r1.FF"
                   ):
        self.reactor_design_spec_file_name = reactor_design_spec_file_name
        self.cross_section_library_file_name = cross_section_library_file_name
        self.form_function_library_file_name = form_function_library_file_name

        print(f"{self.__class__.__name__} instance has been successfully initialized.")
        print(f"  - Reactor Design Spec File: {self.reactor_design_spec_file_name}")
        print(f"  - Cross Section Library File: {self.cross_section_library_file_name}")
        print(f"  - Form Function Library File: {self.form_function_library_file_name}")

        # dictionary
        self.batch_id_composition_dictionary = None
        self.composition_2g_macxs_dictionary = None
    
    # AstraResultParser의 객체의 메서드인 
        # get_loading_pattern_and_batch_ids의 두번째 반환값인
            # self.all_authentic_batch_ids를 받아와서
                # 1) batchID <-> composition 딕셔너리 생성하고
                # 2) composition <-> cross-section 딕셔너리 생성해서 저장 

    # 크로스섹션에서 필요한 부분을 추출하는 부분

    def build_batch_id_composition_dictionary(self):
        """
        Build a dictionary mapping batch ID (e.g., 'A01') to a list of unique composition names found in the corresponding Configuration line.
        The most frequently occurring composition name appears first in the list.
        All compositions included in each batch ID are extracted.
        
        Returns:
            batch_id_composition_dictionary (dict): A dictionary where keys are batch IDs and values are lists of composition names.
        
        Example:
            Example snippet from the reactor design specification file:
            
            Configuration(A01)     = 16gd    BLANKET_220 BLANKET_220  400_NE08W16G 400_NE08W16G    400_NE08W16G 400_NE08W16G    
            Configuration(A02)     = 28gd    BLANKET_220 BLANKET_220  400_NE08W28G 400_NE08W28G    400_NE08W28G 400_NE08W28G  
            Configuration(A03)     = 20gd    BLANKET_220 BLANKET_220  495_NE08W20G 495_NE08W20G    495_NE08W20G 495_NE08W20G   
                
        """

        with open(self.reactor_design_spec_file_name, 'r', encoding='utf-8') as design_spec_file_object:
            lines = design_spec_file_object.readlines()
            batch_id_composition_dictionary = {}

            for line in lines:
                if line.strip().startswith("Configuration"):
                    # Extract batch ID between parentheses
                    key_start = line.find('(')
                    key_end = line.find(')')
                    key = line[key_start + 1 : key_end] # .strip()  # e.g., 'A01', 'A02', etc.

                    # Extract all composition names after '=' (excluding the first two items)
                    equal_index = line.find('=')
                    comps = line[equal_index + 1:].strip().split()
                    composition_candidates = comps[1:]   # '16gd' 이후 모두

                    # 빈 값 걸러내기
                    composition_candidates = [c for c in composition_candidates if c]

                    if composition_candidates:
                        counter = Counter(composition_candidates)
                        # 중복 없이, 가장 많이 등장한 순서대로 리스트 생성
                        most_common = [name for name, _ in counter.most_common()]
                        batch_id_composition_dictionary[key] = most_common
                    else:
                        batch_id_composition_dictionary[key] = []

            print("Function 'build_batch_id_composition_dictionary' executed successfully.")

        self.batch_id_composition_dictionary = batch_id_composition_dictionary # e.g., {'A01': ['BLANKET_220', '400_NE08W16G'], 'A02': ['BLANKET_220', '400_NE08W28G'], ...}
        return batch_id_composition_dictionary



    def xsec_chunk_slicer(self, cross_section_chunk):
        """
        Parses a macroscopic cross-section (MAC section) chunk (list of fixed-width strings) and returns two lists of floating-point values.

        Args:
            cross_section_chunk (list of str): List of 16 strings, each representing a line of fixed-width cross-section data.

        Returns:
            tuple: (list1, list2) - Two lists of floats, corresponding to the first and second group cross-section values.

        Example:
            Example snippet from a cross-section chunk:
            0.735011E-020.734906E-020.734571E-020.733006E-020.731461E-020.728084E-02
            0.724552E-020.720906E-020.717159E-020.713321E-020.709389E-020.705331E-02
            0.701123E-020.696773E-020.692310E-020.687772E-020.683180E-020.678552E-02
            0.673898E-020.669211E-020.664494E-020.659748E-020.654964E-020.650147E-02
            0.645293E-020.640399E-020.635455E-020.630461E-020.625384E-020.620230E-02
            0.615003E-020.609733E-020.604450E-020.599181E-020.593940E-020.588738E-02
            0.583575E-020.578452E-020.573370E-020.548847E-020.525471E-020.503472E-02
            0.483080E-020.464497E-020.447877E-02
            0.126810E+000.126793E+000.126749E+000.127320E+000.128380E+000.130595E+00
            0.132758E+000.134846E+000.136853E+000.138797E+000.140677E+000.142389E+00
            0.143816E+000.144965E+000.145877E+000.146616E+000.147253E+000.147826E+00
            0.148351E+000.148848E+000.149320E+000.149772E+000.150221E+000.150670E+00
            0.151132E+000.151638E+000.152175E+000.152643E+000.152929E+000.152927E+00
            0.152555E+000.151864E+000.150986E+000.150025E+000.149026E+000.148006E+00
            0.146973E+000.145929E+000.144877E+000.139565E+000.134141E+000.128714E+00
            0.123399E+000.118310E+000.113552E+00

        Notes:
            - Each line is parsed in 12-character wide fields.
            - The first 8 lines are for group 1, the next 8 lines are for group 2.
        """

        temp_list_1g = []
        temp_list_2g = []

        # 처음 8줄 처리
        for chunk_line in cross_section_chunk[:8]:
            for i in range(0, len(chunk_line), 12):
                try:
                    value = float(chunk_line[i:i + 12])
                    temp_list_1g.append(value)  # list1에 바로 추가
                except ValueError:
                    print(f"Warning: Could not convert '{chunk_line[i:i+12]}' to float.")
                    pass
        # 다음 8줄 처리
        for chunk_line in cross_section_chunk[8:16]:
            for i in range(0, len(chunk_line), 12):
                try:
                    value = float(chunk_line[i:i + 12])
                    temp_list_2g.append(value)  # list2에 바로 추가
                except ValueError:
                    print(f"Warning: Could not convert '{chunk_line[i:i+12]}' to float.")
                    pass

        return temp_list_1g, temp_list_2g # 1G and 2G macroscopic cross-section lists



    def build_composition_2g_macxs_dictionary(self, all_authentic_batch_ids, batch_id_composition_dictionary = None):
        """
        Builds a dictionary mapping composition names to their 2G macroscopic cross-sections.

        Args:
            all_authentic_batch_ids (list): A list of all authentic batch IDs.
            batch_id_composition_dictionary (dict, optional): A dictionary mapping batch IDs to their composition names. Defaults to None.

        Returns:
            composition_2g_macxs_dictionary (dict): A dictionary where keys are composition names and values are lists of 2G macroscopic cross-sections.
        """

        if batch_id_composition_dictionary is None: # If no dictionary is provided, use the default
            batch_id_composition_dictionary = self.batch_id_composition_dictionary

        composition_name_set = set() # Set to store unique composition names
        
        for batch_id in all_authentic_batch_ids:
            values = batch_id_composition_dictionary.get(batch_id, ["Unknown"]) # Default to ["Unknown"] if batch_id not found
            
            composition_name_set.update(values)
        
        composition_name_set = list(composition_name_set) # e.g., ['BLANKET_220', '400_NE08W16G', '400_NE08W28G', '495_NE08W20G']
        composition_2g_macxs_dictionary = {} # e.g., {'BLANKET_220': [[[], [], [], [], []], [[], [], [], [], []]], '400_NE08W16G': [[[], [], [], [], []], [[], [], [], [], []]], ...}
        #dict_composition_to_2G_MacXS = {}
        
        for name in composition_name_set:
            composition_2g_macxs_dictionary[name] = [[[], # 1G nu-fission
                                                [], #    fission
                                                [], #    capture
                                                [], #    transport
                                                []],#    scattering 
                                                [[], # 2G nu-fission
                                                [], #    fission
                                                [], #    capture
                                                [], #    transport
                                                []]]#    scattering
            
        # with open을 루프보다 먼저 쓰게 되면, 커서 포인터는 파일에서 끝으로 고정되므로
        # 매 루프마다 파일을 다시 열어서, 커서 포인터의 위치를 초기화해줘야함
        for num_comp, comp in enumerate(composition_name_set):

            # 루프마다 담아놓을 임시 list이며 slicer로 1군, 2군 분해한뒤 각각 리스트로 만들고
            # 이후 composition name을 key로, 1군 및 2군의 2차원 리스트 데이터를 value로 하는 딕셔너리 생성 예정
            # 2차원 리스트 가능, "key2": [["a", "b"], ["c", "d", "e"]]

            chunk_mac_nufission_per_loop = []
            chunk_mac_fission_per_loop = []
            chunk_mac_capture_per_loop = []
            chunk_mac_transport_per_loop = []
            chunk_mac_scattering_per_loop = []
            
            with open(self.cross_section_library_file_name, 'r', encoding="utf-8") as xs_file_object:
                #print("iteration number:", num_comp)
                #print("composition name:", comp)
                comp_found = False
                
                while True:
                    line = xs_file_object.readline()
                    if not line:
                        #print("Reached end of file", f"{num_comp}th")
                        #print("Reached end of file", f"{num_comp}{ordinal_suffix(num_comp)}", "\n")
                        break
                    
                    if comp in line.strip():
                        comp_found = True
                        #print(line) # 파일 포인터의 위치는 "COMP 400_5E08W16G"등 comp 명
                    
                    # 아래 부분부터 거시단면적이 시작되므로, chunk list에 추가하면 됨
                    if comp_found and "MAC" in line.strip():
                        #print(line.strip())
                        #print()
                        for _ in range(26):
                        #print(f_bchA.readline())
                            xs_file_object.readline().strip()

                        # (1) 현재 loop에서 composition name에 해당하는 MAC xsec 수집 (chunk 형태)

                        # 수정 필요,, 지금은 마지막 루프의 comp만 추가하는듯, comp 명에 따른 key-value 쌍으로 저장할 필요
                        # nu-fission chunk
                        for _ in range(16):
                            chunk_mac_nufission_per_loop.append(xs_file_object.readline().strip())
                        # fission chunk
                        for _ in range(16):
                            chunk_mac_fission_per_loop.append(xs_file_object.readline().strip())
                        # capture chunk
                        for _ in range(16):
                            chunk_mac_capture_per_loop.append(xs_file_object.readline().strip())
                        # transport chunk
                        for _ in range(16):
                            chunk_mac_transport_per_loop.append(xs_file_object.readline().strip())
                        # scattering chunk
                        for _ in range(16):
                            chunk_mac_scattering_per_loop.append(xs_file_object.readline().strip())
                        

                        # (2) loop마다 수집한 chunk 데이터들을 slicing한 뒤 각각 1군 2군 리스트로 변환
                        
                        # MAC xsec, sliced
                        sliced_mac_nufission_1g, sliced_mac_nufission_2g = self.xsec_chunk_slicer(chunk_mac_nufission_per_loop)
                        sliced_mac_fission_1g, sliced_mac_fission_2g = self.xsec_chunk_slicer(chunk_mac_fission_per_loop)
                        sliced_mac_capture_1g, sliced_mac_capture_2g = self.xsec_chunk_slicer(chunk_mac_capture_per_loop)
                        sliced_mac_transport_1g, sliced_mac_transport_2g = self.xsec_chunk_slicer(chunk_mac_transport_per_loop)
                        sliced_mac_scattering_1g, sliced_mac_scattering_2g = self.xsec_chunk_slicer(chunk_mac_scattering_per_loop)

                        # (3) 딕셔너리화, 1군 2군 list는 해당되는 comp명에 대해 2차원의 list로 만든 후 value로 덧붙임

                        # 1G MAC XS
                        composition_2g_macxs_dictionary[comp][0][0].extend(sliced_mac_nufission_1g)
                        composition_2g_macxs_dictionary[comp][0][1].extend(sliced_mac_fission_1g)
                        composition_2g_macxs_dictionary[comp][0][2].extend(sliced_mac_capture_1g)
                        composition_2g_macxs_dictionary[comp][0][3].extend(sliced_mac_transport_1g)
                        composition_2g_macxs_dictionary[comp][0][4].extend(sliced_mac_scattering_1g)

                        # 2G MAC XS
                        composition_2g_macxs_dictionary[comp][1][0].extend(sliced_mac_nufission_2g)
                        composition_2g_macxs_dictionary[comp][1][1].extend(sliced_mac_fission_2g)
                        composition_2g_macxs_dictionary[comp][1][2].extend(sliced_mac_capture_2g)
                        composition_2g_macxs_dictionary[comp][1][3].extend(sliced_mac_transport_2g)
                        composition_2g_macxs_dictionary[comp][1][4].extend(sliced_mac_scattering_2g)

                    if comp_found and "DEL2" in line.strip(): # If "DEL2" is found, stop processing
                        #print(line)
                        comp_found = False
                        break
        print("Function 'dict_composition_to_2G_MacXS_constructor' executed successfully.")
        
        self.composition_2g_macxs_dictionary = composition_2g_macxs_dictionary
        return composition_2g_macxs_dictionary # e.g., {'BLANKET_220': [[[], [], [], [], []], [[], [], [], [], []]], '400_NE08W16G': [[[], [], [], [], []], [[], [], [], [], []]], ...}


class MacXSMatrixBuilder():
    # 2D LP인지 3D LP인지 어떻게 입력받지
    def __init__(self):
        self.num_rows_in_quarter_lp = 5

        print(f"{self.__class__.__name__} instance has been successfully initialized.")
        print(f"  - Number of rows in the assembly-wise quadrant loading pattern: {self.num_rows_in_quarter_lp}")

    def linear_interpolation(self, x, x0, x1, y0, y1):
        """1-D linear interpolation function"""
        return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0)

    # HACK: 해당 메서드는 연소도에 대한 보간 기능 없이, DB에 작성된 BOC기준 XS를 이용함. 2차원 반경방향 연소도는 존재하지만, 2차원 연소도를 이용해 보간이 가능할지는 불확실함
    def build_macxs_3d_matrix(self,
                              lpd_matrix_authentic_batchID,
                              # assembly_wise_burnup_2d_matrix
                              batch_id_composition_dictionary,
                              composition_2g_macxs_dictionary
                              ):
        """
        Build a 3D macroscopic cross-section (MAC XS) matrix based on the given loading pattern and batch-composition mapping.
        method for 2D core loading patterns, specifically for a quadrant loading pattern.
        
        Args:
            lpd_matrix_authentic_batchID (2D list): Core layout matrix containing batch IDs
            batch_id_composition_dictionary (dict): Mapping of batch IDs to their composition names
            composition_2g_macxs_dictionary (dict): Mapping of composition names to their 2G MAC XS data
            
        Returns:
            mac_xs_matrix3d(np.ndarray): A 3D numpy array of shape (5, 5, 10) representing the macroscopic cross-section matrix.

        """
        
        print("NOTE: Macroscopic XS matrix for a \'2-D Quadrant Loading Patterns\'")
        print("NOTE: This method can **only** generate the XS matrix at the BOC (Beginning of Cycle). - 25-06-27")
 
        
        # HACK: (하드코딩 부분)
        # 각 batch ID에 대한 값이 리스트로 되어 있으므로, get(key, [key])[0]을 사용해 리스트의 첫 번째 구성요소만 사용함
        # key가 없을 경우에는 기본값으로 [key]를 넣고, 역시 첫 번째 요소로 원래 key 값을 유지
        
        # 순서: lpd_batchID 2D matrix -> composition 2D matrix 생성 -> 
        composition_matrix2d = np.array(lpd_matrix_authentic_batchID).astype(object) # 빈 array 초기화
        vectorized_lookup = np.vectorize(
            lambda key: batch_id_composition_dictionary.get(key, [key])[0]
        )
        composition_matrix2d = vectorized_lookup(lpd_matrix_authentic_batchID)
        
        mac_xs_matrix3d = np.zeros((5, 5, 10))
        
        dim = self.num_rows_in_quarter_lp
        
        for i in range(dim):
            for j in range(dim):

                comp_name_per_cell = composition_matrix2d[i, j]

                # comp_name_per_3d_cell이 0, '0', None이면 그대로 0 유지
                if comp_name_per_cell in [0, '0', None]:
                    continue  # 이미 mac_xs_matrix4d가 0이므로 연산 필요 없음
                
                # shape : (24, 5, 5), composition별로 가지는 모든 MAC XS
                comp_xs_data = composition_2g_macxs_dictionary.get(comp_name_per_cell)
                np_comp_xs_data = np.array(comp_xs_data)
                print("shape of comp_xs_data", np_comp_xs_data.shape ) # 현재는 (2, 5, 45)임 아마 2는 fast/thermal이고, 5는 
                if comp_xs_data is None:
                    print(f"No macroscopic cross-section data available: Composition '{comp_name_per_cell}' does not exist.")
                    break
                temp_xs_list = []

                # 1G 및 2G의 5개 단면적 값 (총 10개)
                for group in range(2):  # 0: 1G, 1: 2G
                    for xs_idx in range(5):  # 0~4
                        xs_value = comp_xs_data[group][xs_idx][0]
                        temp_xs_list.append(xs_value)

                        np_temp_xs_list = np.array(temp_xs_list)
                        print("shape of temp_xs_list", np_temp_xs_list.shape)

                mac_xs_matrix3d[i][j] = np.array(temp_xs_list)

        return mac_xs_matrix3d

    def build_macxs_4d_matrix(self,
                              lpd_matrix_authentic_batchID, # AstraResultParser
                              assembly_wise_burnup_3d_matrix,  # AstraResultParser
                              batch_id_composition_dictionary, # AstraDBParser
                              composition_2g_macxs_dictionary, # AstraDBParser
                              ):
        """
        Build a 4D macroscopic cross-section (MAC XS) matrix based on the given burnup matrix and batch-composition mapping.

        This function creates a (24, 5, 5, 10) macroscopic XS matrix used for core simulation. 
        The 3D core is divided axially into 25 layers: 
        - The topmost layer (cutback region) is assumed to be composed entirely of blanket material ("BLANKET_400").
        - The remaining 24 layers are filled by repeating a single composition matrix derived from batch ID mappings.

        Composition per batch is assumed to be a list, and only the first element of that list is used.
        This design assumes a simplified case for A-type batches and may need to be generalized in the future.

        Parameters:
            lpd_matrix_authentic_batchID (2D list): Core layout matrix containing batch IDs.
            assembly_wise_burnup_3d_matrix (3D list): Axial-wise burnup values for each cell in the core.
            batch_id_composition_dictionary (dict): Mapping from batch ID to a list of composition names. Only the first element is used.
            composition_2g_macxs_dictionary (dict): Dictionary containing 2-group MAC XS data for each composition.

        Returns:
            np.ndarray: A 4D numpy array of shape (24, 5, 5, 10) representing the macroscopic cross-section matrix.

        Notes:
            - Only the first composition from each batch's list is used.
            - Composition matrix is reused across all 24 non-blanket layers.
            - This function is not yet generalized for non-A-type batch layouts.
        """
        
        print("NOTE: Macroscopic XS matrix for a \'3-D Quadrant Core\'")
        
        # burnup index: burnup matrix를 받은 후 연소도에 따라 단면적(xs)을 보간하기 위한 용도
        burnup_level_idx = [0.000000E+00, 0.100000E+02, 0.500000E+02, 0.500000E+03, 0.100000E+04, 0.200000E+04,
                            0.300000E+04, 0.400000E+04, 0.500000E+04, 0.600000E+04, 0.700000E+04, 0.800000E+04,
                            0.900000E+04, 0.100000E+05, 0.110000E+05, 0.120000E+05, 0.130000E+05, 0.140000E+05,
                            0.150000E+05, 0.160000E+05, 0.170000E+05, 0.180000E+05, 0.190000E+05, 0.200000E+05,
                            0.210000E+05, 0.220000E+05, 0.230000E+05, 0.240000E+05, 0.250000E+05, 0.260000E+05,
                            0.270000E+05, 0.280000E+05, 0.290000E+05, 0.300000E+05, 0.310000E+05, 0.320000E+05,
                            0.330000E+05, 0.340000E+05, 0.350000E+05, 0.400000E+05, 0.450000E+05, 0.500000E+05,
                            0.550000E+05, 0.600000E+05, 0.650000E+05]
        
        # 0. blanket matrirx2d 구성 필요
        # !!! dict_composition_to_2G_MacXS_constructor()에서 blanket에 대한 정보를 key value 쌍의 dictionary로 사전 구성
        blanket_400_matrix2d = np.array([['BLANKET_400']*self.num_rows_in_quarter_lp,
                                    ['BLANKET_400']*self.num_rows_in_quarter_lp,
                                    ['BLANKET_400']*self.num_rows_in_quarter_lp,
                                    ['BLANKET_400']*(self.num_rows_in_quarter_lp-1) + ['0'],
                                    ['BLANKET_400']*(self.num_rows_in_quarter_lp-2) + ['0', '0']], dtype=object)
        
        blanket_220_matrix2d = np.array([['BLANKET_220']*self.num_rows_in_quarter_lp,
                                    ['BLANKET_220']*self.num_rows_in_quarter_lp,
                                    ['BLANKET_220']*self.num_rows_in_quarter_lp,
                                    ['BLANKET_220']*(self.num_rows_in_quarter_lp-1) + ['0'],
                                    ['BLANKET_220']*(self.num_rows_in_quarter_lp-2) + ['0', '0']], dtype=object)

         # 1. composition matrix2d 생성하는 부분
            # np.vectorize를 사용하여 딕셔너리 조회
            # shape : (5, 5)
        composition_matrix2d = np.array(lpd_matrix_authentic_batchID).astype(object) # 빈 array 초기화
        
        # HACK: (하드코딩 부분)
        # 각 batch ID에 대한 값이 리스트로 되어 있으므로, get(key, [key])[0]을 사용해 리스트의 첫 번째 구성요소만 사용함
        # key가 없을 경우에는 기본값으로 [key]를 넣고, 역시 첫 번째 요소로 원래 key 값을 유지
        
        vectorized_lookup = np.vectorize(
            lambda key: batch_id_composition_dictionary.get(key, [key])[0]
        )
        composition_matrix2d = vectorized_lookup(lpd_matrix_authentic_batchID)
        
        # 2. composition matrix3d 생성부
            # 그 전에 필요한것: 연소도에 따른 blanket의 거시단면적 딕셔너리 (blanket220, blanket400), blanket matrix2d
            # 250221) 우선 2D matrix 이용,  3D (24,5,5) 형태로 composition matrix3d를 만든 후 3번 섹션에서 변환
            # shape : (24, 5, 5)
            
        composition_matrix3d = np.vstack([
            blanket_400_matrix2d[np.newaxis, :, :],  # (1,5,5) -> 첫 번째 층
            np.tile(composition_matrix2d, (23,1,1))  # (23,5,5) -> 나머지 23개 층
        ])
        # 3. (composition_matrix3D + burnup_matrix3D) -> macroscopic matrix 4D
            # shape : (24,5,5,10)
        mac_xs_matrix4d = np.zeros((24, 5, 5, 10))

        (dim0, dim1, dim2) = np.array(assembly_wise_burnup_3d_matrix).shape

        for i in range(dim0):
                for j in range(dim1):
                    for k in range(dim2):
                        
                        # 현재 cell의 연소도 값
                        x = assembly_wise_burnup_3d_matrix[i][j][k]

                        # burnup_level_idx에서 가장 가까운 두 개의 인덱스를 찾음
                        temp_difference_list = [(abs(x - burnup_level_idx[idx]), idx) for idx in range(len(burnup_level_idx))]
                        temp_difference_list.sort()

                        temp_min_idx_previous = temp_difference_list[0][1]
                        temp_min_idx_next = temp_difference_list[1][1]

                        # x0, x1 설정
                        x0 = burnup_level_idx[temp_min_idx_previous]
                        x1 = burnup_level_idx[temp_min_idx_next]

                        # 현재 cell의 조성 정보 가져오기
                        comp_name_per_3d_cell = composition_matrix3d[i, j, k]

                        # comp_name_per_3d_cell이 0, '0', None이면 그대로 0 유지
                        if comp_name_per_3d_cell in [0, '0', None]:
                            continue  # 이미 mac_xs_matrix4d가 0이므로 연산 필요 없음
                        
                        # shape : (24, 5, 5), composition별로 가지는 모든 MAC XS
                        comp_xs_data = composition_2g_macxs_dictionary.get(comp_name_per_3d_cell)
                        if comp_xs_data is None:
                            print(f"No macroscopic cross-section data available: Composition '{comp_name_per_3d_cell}' does not exist.")
                            break

                        # 보간된 값을 저장할 리스트
                        temp_xs_list = []

                        # 1G 및 2G의 5개 단면적 값 (총 10개)
                        for group in range(2):  # 0: 1G, 1: 2G
                            for xs_idx in range(5):  # 0~4
                                interpolated_value = self.linear_interpolation(
                                    x, x0, x1,
                                    comp_xs_data[group][xs_idx][temp_min_idx_previous],
                                    comp_xs_data[group][xs_idx][temp_min_idx_next]
                                )
                                temp_xs_list.append(interpolated_value)

                        # MAC XS 10개 값을 mac_xs_matrix4d의 하나의 cell에 저장
                        mac_xs_matrix4d[i][j][k] = np.array(temp_xs_list)

        print("Function 'build_macxs_4d_matrix' executed successfully.")
        return mac_xs_matrix4d
    


"""
호출 흐름 예시
1. AstraResultParser 클래스

    parser = AstraResultParser("ASTRA", "your_result_file.txt")
    lpd_matrix, all_authentic_batch_ids = parser.get_loading_pattern_and_batch_ids()
    burnup_matrix3d = parser.get_assembly_3d_distribution("-B3D", step_index=0)

2. AstraDBParser 클래스

    db = AstraDBParser()
    batch_dict = db.build_batch_id_composition_dictionary()
    macxs_dict = db.build_composition_2g_macxs_dictionary(all_authentic_batch_ids, batch_dict)

3. MacXSMatrixBuilder 클래스

    builder = MacXSMatrixBuilder()
    mac_xs_4d = builder.build_macxs_4d_matrix(
        lpd_matrix,
        burnup_matrix3d,
        batch_dict,
        macxs_dict
    )

"""

parser = AstraResultParser("ASTRA", "your_result_file.txt")
lpd_matrix, fuel_batch_ids = parser.get_loading_pattern_and_batch_ids()
burnup_matrix3d = parser.get_assembly_3d_distribution("-B3D", step_index=0)


db = AstraDBParser()
batch_dict = db.build_batch_id_composition_dictionary()
macxs_dict = db.build_composition_2g_macxs_dictionary(fuel_batch_ids, batch_dict)