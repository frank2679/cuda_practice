import subprocess
import pytest

@pytest.mark.parametrize("binary", [
                                    './reduce'
                                    ])
@pytest.mark.parametrize("argument", ['1', '3', '5', '8', '10', '11', '12', '14', '17', '20', '21', '30'])
def test_cpp_binary(binary, argument):
    # Replace 'binary' with the name of your binary file
    # Run the binary with the argument and capture the output
    output = subprocess.check_output([binary, argument]).decode('utf-8')
    # Set the expected output to 'Case passed.' when the binary returns EXIT_SUCCESS
    expected_output = 'Case passed.' if subprocess.call([binary, argument]) == 0 else ''
    # Assert that the output matches the expected output
    assert (output.find(expected_output) != -1)
