@echo off
setlocal enabledelayedexpansion

:: 设置要使用的算法
set algorithms=all

:: 设置要执行的 hash_size 列表
set hash_sizes=4 8 16 32 64 128 256 1024

:: 指定输出文件
set output_file=output.txt

:: 清空输出文件（如果存在）
type nul > %output_file%

:: 遍历每个 hash_size 并执行命令
for %%s in (%hash_sizes%) do (
    echo Executing for hash_size=%%s... >> %output_file%
    echo Executing for hash_size=%%s...
    python g:\trae\myhash\evaluate_hashes.py --dataset scid --algorithms %algorithms% --hash_size %%s >> %output_file% 2>&1
    type %output_file%
)

echo All tasks completed. >> %output_file%
echo All tasks completed.

pause