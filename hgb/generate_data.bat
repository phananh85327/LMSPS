@echo off

call E:\miniconda3\Scripts\activate.bat
call conda activate lmsps_env

call python amazon_extract.py --input ..\downloads\amazon-meta.txt.gz --output ..\data\AMAZON_filtered
call python dat_generator.py --input ..\data\AMAZON_filtered --output ..\data\AMAZON --verify
call python audit_amazon_data.py --filtered ..\data\AMAZON_filtered --out ..\data\AMAZON\audit.json