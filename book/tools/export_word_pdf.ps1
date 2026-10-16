param(
    [Parameter(Mandatory = $true)][string]$DocumentPath,
    [Parameter(Mandatory = $true)][string]$PdfPath
)

$ErrorActionPreference = 'Stop'
$wordInstance = $null
$openedDocument = $null
try {
    $wordInstance = New-Object -ComObject Word.Application
    $wordInstance.Visible = $false
    $wordInstance.DisplayAlerts = 0
    $wordInstance.AutomationSecurity = 3
    $openedDocument = $wordInstance.Documents.Open($DocumentPath, $false, $true)
    $openedDocument.Repaginate()
    $openedDocument.ExportAsFixedFormat($PdfPath, 17)
    Write-Output "Exported $PdfPath"
}
finally {
    $doNotSave = 0
    if ($null -ne $openedDocument) {
        $openedDocument.Close([ref]$doNotSave)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($openedDocument)
    }
    if ($null -ne $wordInstance) {
        $wordInstance.Quit([ref]$doNotSave)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordInstance)
    }
}
