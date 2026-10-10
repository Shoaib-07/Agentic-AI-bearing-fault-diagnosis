Set-Location $PSScriptRoot
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -m streamlit run dashboard/app.py
} else {
    python -m streamlit run dashboard/app.py
}
