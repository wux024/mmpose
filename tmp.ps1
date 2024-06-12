# 设置目标根目录及其它变量
$rootPath = "D:\wux024\mmpose\configs\animal_2d_keypoint\bottomup"
$searchString = "batch_size=32"
$replacement = "batch_size=1"
$count = 0

# 定义函数用于计数并替换第二次和第三次出现的字符串
function ReplaceSecondAndThirdOccurrence($content) {
    $global:count = 0
    return $content -replace $searchString, { 
        if (++$global:count -ge 2 -and $global:count -le 3) { $replacement } else { $args[0] }
    }
}

# 主处理逻辑
$subDirs = Get-ChildItem -Path $rootPath -Directory

foreach ($subDir in $subDirs) {
    $currentSubDirPath = $subDir.FullName
    $pyFiles = Get-ChildItem -Path $currentSubDirPath -Filter "*.py" -File
    
    foreach ($pyFile in $pyFiles) {
        $filePath = $pyFile.FullName
        $content = Get-Content -Path $filePath -Raw
        $newContent = ReplaceSecondAndThirdOccurrence $content
        
        # 确保实际修改文件内容（移除-WhatIf以应用更改）
        Set-Content -Path $filePath -Value $newContent
        
        Write-Host "Processed file: $filePath"
    }
}