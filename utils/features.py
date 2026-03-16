import pandas as pd

def getCorrelation(df: pd.DataFrame, method: str):
    numericCols = [
        col for col in df.select_dtypes(include="number").columns
    ]
    # Checking Required as .cor::method exepects Literal["", ..]
    if method in ('pearson', 'kendall', 'spearman'):
        return (
            df[numericCols]
            .corr(method=method)
            .round(4) * 100
        )