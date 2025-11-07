"""Medienos atliekų analizatorius API.

Ši programa skirta apdoroti gamybos (pjovimo/apdirbimo) duomenis ir
apskaičiuoti pagrindinius rodiklius: žaliavos praradimus, dažniausias
priežastis bei daugiausia nuostolių sukeliančius gaminius. Taip pat
numatyta bazinė integracijos sąsaja su vaizdo atpažinimo sistema.
"""
from __future__ import annotations

import csv
from datetime import date
from io import BytesIO
from typing import Dict, Iterable, List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Duomenų modeliai
# --------------------------------------------------------------------------------------


class ProductionRecord(BaseModel):
    """Vienos gamybos operacijos įrašas."""

    batch_id: Optional[str] = Field(
        default=None, description="Unikalus gamybos partijos identifikatorius."
    )
    operation: Optional[str] = Field(
        default=None, description="Operacijos pavadinimas (pjovimas, šlifavimas ir pan.)."
    )
    raw_input_kg: float = Field(
        ..., ge=0, description="Į gamybą paduotos žaliavos kiekis kilogramais."
    )
    finished_output_kg: float = Field(
        ..., ge=0, description="Pagamintas tinkamos produkcijos kiekis kilogramais."
    )
    waste_kg: Optional[float] = Field(
        default=None,
        ge=0,
        description="Atliekų kiekis kilogramais (jei nenurodyta – apskaičiuojama automatiškai).",
    )
    defect_reason: Optional[str] = Field(
        default=None, description="Dažniausia atliekos/defekto priežastis."
    )
    product_code: Optional[str] = Field(
        default=None, description="Detalės ar gaminio kodas/pavadinimas."
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="Operacijos data/laikas ISO formatu (naudojama chronologinei analizei).",
    )


class WasteBreakdown(BaseModel):
    """Sugrupuoti atliekų rodikliai."""

    label: str
    waste_kg: float
    share: float
    count: int


class TimelinePoint(BaseModel):
    """Atliekų dinamika laike."""

    date: date
    waste_kg: float
    raw_input_kg: float


class AnalysisSummary(BaseModel):
    """Bendras medienos atliekų apibendrinimas."""

    total_batches: int
    total_raw_input_kg: float
    total_finished_output_kg: float
    total_waste_kg: float
    waste_ratio: float
    estimated_loss_eur: Optional[float] = None


class AnalyzeResponse(BaseModel):
    """Pilnas analizės atsakymas."""

    summary: AnalysisSummary
    waste_by_reason: List[WasteBreakdown]
    waste_by_product: List[WasteBreakdown]
    suggestions: List[str]
    timeline: List[TimelinePoint] = Field(default_factory=list)


class VisualEstimationRequest(BaseModel):
    """Paprastas vaizdo analizatoriaus įvesties modelis."""

    waste_pixels: int = Field(..., ge=0, description="Atliekų plotas pikseliais (iš ML modelio maskės).")
    total_pixels: int = Field(..., gt=0, description="Bendras analizuojamo vaizdo plotas pikseliais.")
    max_mass_kg: Optional[float] = Field(
        default=None,
        ge=0,
        description="Kalibracijos parametras – kiek kg atliekų atitinka visas kadro plotas.",
    )


class VisualEstimationResponse(BaseModel):
    """Vaizdo analizatoriaus atsakymo modelis."""

    waste_ratio: float
    estimated_waste_kg: Optional[float] = None


# --------------------------------------------------------------------------------------
# Pagalbinės konstantos ir žemėlapiai
# --------------------------------------------------------------------------------------


REQUIRED_BASE_COLUMNS = {"raw_input_kg", "finished_output_kg"}
COLUMN_ALIASES: Dict[str, Iterable[str]] = {
    "raw_input_kg": {"raw_input", "raw_material_kg", "žaliava_kg", "input_kg"},
    "finished_output_kg": {
        "finished_output",
        "good_output_kg",
        "pagaminta_kg",
        "good_qty_kg",
    },
    "waste_kg": {"scrap_kg", "waste", "nuostoliai_kg"},
    "defect_reason": {"reason", "defect", "priezastis", "priežastis"},
    "product_code": {"product", "item", "gaminiokodas", "detale"},
    "timestamp": {"date", "datetime", "laikas"},
}
DEFAULT_TARGET_WASTE_RATIO = 0.05


# --------------------------------------------------------------------------------------
# Duomenų įkėlimo ir apdorojimo funkcijos
# --------------------------------------------------------------------------------------


def detect_delimiter(data: bytes) -> str:
    """Aptinka CSV skyriklį pagal pateiktą baitų pavyzdį."""

    sample = data[:1024].decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def read_dataframe_from_bytes(data: bytes) -> pd.DataFrame:
    """Įkelia CSV duomenis nepriklausomai nuo skyriklio."""

    if not data:
        raise HTTPException(status_code=400, detail="Pateiktas tuščias failas.")

    delimiter = detect_delimiter(data)
    try:
        return pd.read_csv(BytesIO(data), delimiter=delimiter)
    except Exception as exc:  # pragma: no cover - detalus pranešimas
        raise HTTPException(status_code=400, detail=f"Nepavyko nuskaityti CSV: {exc}") from exc


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sulygina stulpelių pavadinimus ir taiko sinonimų žemėlapius."""

    renamed = {
        column: column.strip().lower().replace(" ", "_") for column in df.columns
    }
    df = df.rename(columns=renamed)

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                df = df.rename(columns={alias: canonical})
                break

    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Trūksta privalomų stulpelių: {', '.join(sorted(missing))}.",
        )

    if "waste_kg" not in df.columns:
        df["waste_kg"] = (df["raw_input_kg"] - df["finished_output_kg"]).clip(lower=0)

    numeric_columns = ["raw_input_kg", "finished_output_kg", "waste_kg"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["defect_reason"] = (
        df.get("defect_reason", pd.Series(index=df.index, dtype="object"))
        .fillna("Nežinoma priežastis")
        .astype(str)
    )
    df["product_code"] = (
        df.get("product_code", pd.Series(index=df.index, dtype="object"))
        .fillna("Nežinomas gaminys")
        .astype(str)
    )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "batch_id" not in df.columns:
        df["batch_id"] = pd.Series(range(1, len(df) + 1), index=df.index).astype(str)

    return df


def build_breakdown(df: pd.DataFrame, column: str, total_waste: float) -> List[WasteBreakdown]:
    """Sugeneruoja atliekų suskaidymą pagal nurodytą stulpelį."""

    grouped = (
        df.groupby(column)["waste_kg"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    )

    breakdown: List[WasteBreakdown] = []
    for label, row in grouped.iterrows():
        waste = float(row["sum"])
        share = float(waste / total_waste) if total_waste else 0.0
        breakdown.append(
            WasteBreakdown(
                label=str(label), waste_kg=round(waste, 3), share=round(share, 4), count=int(row["count"])
            )
        )
    return breakdown


def build_timeline(df: pd.DataFrame) -> List[TimelinePoint]:
    """Paruošia atliekų dinamiką pagal dienas."""

    if "timestamp" not in df.columns:
        return []

    timeline_df = df.dropna(subset=["timestamp"])
    if timeline_df.empty:
        return []

    grouped = timeline_df.groupby(pd.Grouper(key="timestamp", freq="D")).agg(
        waste_kg=("waste_kg", "sum"), raw_input_kg=("raw_input_kg", "sum")
    )

    points: List[TimelinePoint] = []
    for ts, row in grouped.iterrows():
        if pd.isna(ts):
            continue
        points.append(
            TimelinePoint(
                date=ts.date(),
                waste_kg=round(float(row["waste_kg"]), 3),
                raw_input_kg=round(float(row["raw_input_kg"]), 3),
            )
        )
    return points


def create_suggestions(
    summary: AnalysisSummary,
    waste_by_reason: List[WasteBreakdown],
    waste_by_product: List[WasteBreakdown],
    target_ratio: float,
) -> List[str]:
    """Sugeneruoja trumpas rekomendacijas remiantis rezultatais."""

    suggestions: List[str] = []

    if summary.waste_ratio > target_ratio:
        suggestions.append(
            "Atliekų lygis ({:.1%}) viršija tikslą ({:.1%}). Rekomenduojama atlikti proceso auditą.".format(
                summary.waste_ratio, target_ratio
            )
        )

    if waste_by_reason:
        top_reason = waste_by_reason[0]
        suggestions.append(
            "Dažniausia nuostolių priežastis – {} ({:.2f} kg). Peržiūrėkite operatorių pastabas.".format(
                top_reason.label, top_reason.waste_kg
            )
        )

    if waste_by_product:
        top_product = waste_by_product[0]
        suggestions.append(
            "Daugiausia atliekų generuoja gaminys {} ({:.2f} kg). Įvertinkite brėžinius ir įrankius.".format(
                top_product.label, top_product.waste_kg
            )
        )

    if summary.estimated_loss_eur:
        suggestions.append(
            "Žaliavos nuostolių vertė siekia apie {:.2f} €. Palyginkite su gaminio pelningumu.".format(
                summary.estimated_loss_eur
            )
        )

    if not suggestions:
        suggestions.append("Atliekų lygis atitinka tikslus – reikšmingų problemų nenustatyta.")

    return suggestions


def analyze_dataframe(
    df: pd.DataFrame, target_ratio: float, unit_cost_eur: float
) -> AnalyzeResponse:
    """Atlieka pagrindinius skaičiavimus ir suformuoja atsakymą."""

    total_raw = float(df["raw_input_kg"].sum())
    total_finished = float(df["finished_output_kg"].sum())
    total_waste = float(df["waste_kg"].sum())
    waste_ratio = float(total_waste / total_raw) if total_raw else 0.0

    estimated_loss = float(total_waste * unit_cost_eur) if unit_cost_eur else None

    summary = AnalysisSummary(
        total_batches=int(len(df)),
        total_raw_input_kg=round(total_raw, 3),
        total_finished_output_kg=round(total_finished, 3),
        total_waste_kg=round(total_waste, 3),
        waste_ratio=round(waste_ratio, 4),
        estimated_loss_eur=round(estimated_loss, 2) if estimated_loss is not None else None,
    )

    waste_by_reason = build_breakdown(df, "defect_reason", total_waste)
    waste_by_product = build_breakdown(df, "product_code", total_waste)
    timeline = build_timeline(df)
    suggestions = create_suggestions(summary, waste_by_reason, waste_by_product, target_ratio)

    return AnalyzeResponse(
        summary=summary,
        waste_by_reason=waste_by_reason,
        waste_by_product=waste_by_product,
        suggestions=suggestions,
        timeline=timeline,
    )


# --------------------------------------------------------------------------------------
# Vaizdo analizatoriaus pagalbinė klasė
# --------------------------------------------------------------------------------------


class VisualWasteEstimator:
    """Paprastas atliekų įverčio skaičiuotuvas iš ML modelio rezultatų.

    Tikslioje integracijoje čia būtų naudojami kompiuterinės regos moduliai
    (pvz., `opencv-python`, `torch`, `onnxruntime`). Šis įrankis apdoroja
    tik bazinius skaitinius signalus iš modelio (atliekų maskę) ir paverčia
    juos į proporciją bei galimą masę kilogramais.
    """

    def __init__(self, calibration_constant: float = 1.0) -> None:
        self.calibration_constant = calibration_constant

    def estimate_waste_ratio(self, waste_pixels: int, total_pixels: int) -> float:
        if total_pixels <= 0:
            return 0.0
        ratio = (waste_pixels / total_pixels) * self.calibration_constant
        return float(max(0.0, min(ratio, 1.0)))

    def estimate_waste_mass(self, waste_ratio: float, max_mass_kg: Optional[float]) -> Optional[float]:
        if max_mass_kg is None:
            return None
        return float(max(0.0, waste_ratio * max_mass_kg))


visual_estimator = VisualWasteEstimator()


# --------------------------------------------------------------------------------------
# FastAPI aplikacija ir maršrutai
# --------------------------------------------------------------------------------------


app = FastAPI(
    title="Medienos atliekų analizatorius",
    description="API, skirta medienos apdirbimo atliekų rodiklių skaičiavimui ir vizualizacijai.",
    version="1.0.0",
)


@app.get("/")
def root() -> Dict[str, str]:
    """Sveikatos patikra / pagrindinė informacija."""

    return {
        "service": "Medienos atliekų analizatorius",
        "message": "Naudokite /analyze galutiniam CSV ar JSON duomenų apdorojimui.",
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_file(
    file: UploadFile = File(..., description="CSV failas su gamybos įrašais."),
    target_waste_ratio: float = Form(
        DEFAULT_TARGET_WASTE_RATIO,
        description="Tikslinis atliekų santykis (pvz., 0.05 = 5%).",
    ),
    unit_cost_eur: float = Form(
        0.0,
        description="Vieno kilogramo žaliavos savikaina eurais. Naudojama nuostoliams įvertinti.",
    ),
) -> AnalyzeResponse:
    """Priima CSV failą ir grąžina atliekų analizę."""

    data = await file.read()
    df = read_dataframe_from_bytes(data)
    df = normalize_columns(df)
    return analyze_dataframe(df, target_ratio=target_waste_ratio, unit_cost_eur=unit_cost_eur)


@app.post("/analyze/records", response_model=AnalyzeResponse)
async def analyze_records(
    records: List[ProductionRecord],
    target_waste_ratio: float = DEFAULT_TARGET_WASTE_RATIO,
    unit_cost_eur: float = 0.0,
) -> AnalyzeResponse:
    """Alternatyvus maršrutas, priimantis JSON masyvą su įrašais."""

    if not records:
        raise HTTPException(status_code=400, detail="Nepateikti gamybos duomenys.")

    df = pd.DataFrame([record.dict() for record in records])
    df = normalize_columns(df)
    return analyze_dataframe(df, target_ratio=target_waste_ratio, unit_cost_eur=unit_cost_eur)


@app.post("/visual/estimate", response_model=VisualEstimationResponse)
async def estimate_visual_waste(request: VisualEstimationRequest) -> VisualEstimationResponse:
    """Įvertina atliekų santykį remiantis vaizdo (ML) analize."""

    waste_ratio = visual_estimator.estimate_waste_ratio(
        waste_pixels=request.waste_pixels, total_pixels=request.total_pixels
    )
    estimated_mass = visual_estimator.estimate_waste_mass(
        waste_ratio=waste_ratio, max_mass_kg=request.max_mass_kg
    )
    return VisualEstimationResponse(
        waste_ratio=round(waste_ratio, 4),
        estimated_waste_kg=round(estimated_mass, 3) if estimated_mass is not None else None,
    )


if __name__ == "__main__":  # pragma: no cover - leidžia paleisti lokaliai
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
