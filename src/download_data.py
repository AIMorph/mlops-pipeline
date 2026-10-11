from pathlib import Path

from sklearn.datasets import fetch_openml


def main():
    bunch = fetch_openml("adult", version=2, as_frame=True)
    df = bunch.frame.copy()
    df.columns = [c.replace("-", "_") for c in df.columns]
    df["income"] = (df["class"].astype(str).str.strip() == ">50K").astype(int)
    df = df.drop(columns=["class"])
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(object)
    df = df.sample(n=10000, random_state=42).reset_index(drop=True)

    out = Path("data/raw/adult.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()