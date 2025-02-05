import numpy as np
import datetime


def convert_time_s_in_time_hr_mn_s(time_s):
    time_s_only = int(np.mod(time_s, 60))

    time_hr_mn = np.floor(time_s / 60)
    time_mn = int(np.mod(time_hr_mn, 60))
    time_hr = int(np.floor(time_hr_mn / 60))
    return f"{time_hr:02d}hr:{time_mn:02d}mn:{time_s_only:02d}s"


def get_current_time_in_special_file_name_format():
    """format the current date and time into something like  04m_07d_2022y_08h_06mn """
    current_time = datetime.datetime.now().strftime("%mm_%dd_%Yy_%Hh_%Mmn")
    return current_time
