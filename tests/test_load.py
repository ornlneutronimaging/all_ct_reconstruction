from notebooks.__code.workflow.load import Load
import os

class TestLoad:
    
    def test_keep_only_highest_R_value(self):
        list_images = [
            "/path/to/sample_045_030.tiff",
            "/path/to/sample_045_030_R001.tiff",
            "/path/to/sample_045_030_R002.tiff",
            "/path/to/sample_045_031.tiff",
            "/path/to/sample_045_031_R001.tiff",
            "/path/to/sample_045_032.tiff"
        ]
        expected_output = [
            "/path/to/sample_045_030_R002.tiff",
            "/path/to/sample_045_031_R001.tiff",
            "/path/to/sample_045_032.tiff"
        ]
        o_load = Load()
        output = o_load._keep_only_highest_R_value(list_images)
        print(f"output: {output}")
        assert set(output) == set(expected_output)
        