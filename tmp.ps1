# 设置目标根目录
$rootPath = "D:\wux024\mmpose\configs\animal_2d_keypoint\yoloxpose"

# 获取根目录下的所有子目录
$subDirs = Get-ChildItem -Path $rootPath -Directory

foreach ($subDir in $subDirs) {
    # 构建当前子目录的完整路径
    $currentSubDirPath = $subDir.FullName
    
    # 查找当前子目录下所有的.py文件
    $pyFiles = Get-ChildItem -Path $currentSubDirPath -Filter "*.py" -File
    
    foreach ($pyFile in $pyFiles) {
        # 构建.py文件的完整路径
        $filePath = $pyFile.FullName
        
        # 读取文件内容
        $content = Get-Content -Path $filePath -Raw
        
        # 获取当前子目录的名称
        $dirName = $subDir.Name
        
        # 构建替换后的字符串
        $newContent = $content -replace "macaque-640", "$dirName-640"
        
        # 写入修改后的内容到原文件（这里先不做实际修改，使用-WhatIf预览）
        Set-Content -Path $filePath -Value $newContent
        
        Write-Host "Processed file: $filePath"
    }
}