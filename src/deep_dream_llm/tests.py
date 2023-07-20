"""
A file for tests to ensure code works as expect

TODO This file is a Work in progress
"""

def test_calc_loss():
    assert calc_loss("there","there") == 0.0
    assert calc_loss("hello","there") > 0.0
    try:
        calc_loss("hello there","there")
        print("THis doesn't faile which is weird")
    except:
        print("success")