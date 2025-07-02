import os

def get_data_path(filename="record"):
    """
    獲取資料檔案的絕對路徑。
    假設 'record' 檔案與此腳本在同一個目錄，或在一個子目錄中。
    """
    # 獲取當前腳本所在的目錄的絕對路徑
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # 組合檔案路徑
    # 假設 record 在與腳本同層目錄
    file_path = os.path.join(current_script_dir, filename)

    return file_path

