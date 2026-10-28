# Export docs/final_case_study.docx to PDF with Microsoft Word and report the page count.
# Usage (from the repository root):  powershell -ExecutionPolicy Bypass -File docs/case_study_build/export_pdf.ps1
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$docx = Join-Path $root "docs\final_case_study.docx"
$pdf = Join-Path $root "docs\final_case_study.pdf"
Write-Output "starting Word"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0                                  # wdAlertsNone: never block on a dialog
try {
    Write-Output "opening $docx"
    # FileName, ConfirmConversions, ReadOnly, AddToRecentFiles
    $doc = $word.Documents.Open($docx, $false, $true, $false)
    Write-Output "repaginating"
    $doc.Repaginate()
    $pages = $doc.ComputeStatistics(2)                   # wdStatisticPages
    Write-Output "pages=$pages"
    Write-Output "exporting"
    # SaveAs2 with wdFormatPDF (17); ExportAsFixedFormat can block on some installs.
    $doc.SaveAs2([string]$pdf, 17)
    $doc.Close($false)
    Write-Output "wrote $pdf"
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
