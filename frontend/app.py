
"""Streamlit frontend for the Ames house-price prediction pipeline."""

from __future__ import annotations

import calendar
import os
import sys

import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# 1. Project path and backend imports
# ---------------------------------------------------------

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.main import (
    ENGINEERED_FEATURES,
    category_options,
    default_values,
    load_model,
    market_percentile,
    model_features,
    predict_price,
    reference_price_stats,
)

# Configure the page only once, before other Streamlit UI calls.
st.set_page_config(
    page_title="Haven | House Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# 2. Category labels
# ---------------------------------------------------------

CATEGORY_LABELS = {
    "MSZoning": {
        "A": "Agriculture",
        "C (all)": "Commercial",
        "FV": "Floating village residential",
        "I": "Industrial",
        "RH": "Residential high density",
        "RL": "Residential low density",
        "RP": "Residential low-density park",
        "RM": "Residential medium density",
    },
    "Neighborhood": {
        "Blmngtn": "Bloomington Heights",
        "Blueste": "Bluestem",
        "BrDale": "Briardale",
        "BrkSide": "Brookside",
        "ClearCr": "Clear Creek",
        "CollgCr": "College Creek",
        "Crawfor": "Crawford",
        "Edwards": "Edwards",
        "Gilbert": "Gilbert",
        "IDOTRR": "Iowa DOT and Rail Road",
        "MeadowV": "Meadow Village",
        "Mitchel": "Mitchell",
        "NAmes": "North Ames",
        "NoRidge": "Northridge",
        "NPkVill": "Northpark Villa",
        "NridgHt": "Northridge Heights",
        "NWAmes": "Northwest Ames",
        "OldTown": "Old Town",
        "SWISU": "South/West ISU",
        "Sawyer": "Sawyer",
        "SawyerW": "Sawyer West",
        "Somerst": "Somerset",
        "StoneBr": "Stone Brook",
        "Timber": "Timberland",
        "Veenker": "Veenker",
    },
    "HouseStyle": {
        "1Story": "One story",
        "1.5Fin": "One-and-a-half story, finished",
        "1.5Unf": "One-and-a-half story, unfinished",
        "2Story": "Two story",
        "2.5Fin": "Two-and-a-half story, finished",
        "2.5Unf": "Two-and-a-half story, unfinished",
        "SFoyer": "Split foyer",
        "SLvl": "Split level",
    },
    "BldgType": {
        "1Fam": "Single-family detached",
        "2fmCon": "Two-family conversion",
        "Duplex": "Duplex",
        "Twnhs": "Townhouse",
        "TwnhsE": "Townhouse end unit",
    },
    "GarageType": {
        "2Types": "More than one garage type",
        "Attchd": "Attached garage",
        "Basment": "Basement garage",
        "BuiltIn": "Built-in garage",
        "CarPort": "Carport",
        "Detchd": "Detached garage",
        "None": "No garage",
    },
    "GarageFinish": {
        "Fin": "Finished",
        "RFn": "Rough finished",
        "Unf": "Unfinished",
        "None": "No garage",
    },
    "SaleType": {
        "WD": "Normal warranty deed",
        "CWD": "Cash warranty deed",
        "VWD": "VA warranty deed",
        "New": "New home",
        "COD": "Court officer deed",
        "Con": "Contract",
        "ConLw": "Low down-payment contract",
        "ConLI": "Low-interest contract",
        "ConLD": "Low down-payment/interest contract",
        "Oth": "Other",
    },
    "SaleCondition": {
        "Normal": "Normal sale",
        "Abnorml": "Abnormal sale",
        "AdjLand": "Adjoining land purchase",
        "Alloca": "Allocation sale",
        "Family": "Sale between family members",
        "Partial": "Partial or unfinished home sale",
    },
    "CentralAir": {
        "Y": "Yes",
        "N": "No",
    },
    "PavedDrive": {
        "Y": "Paved",
        "P": "Partially paved",
        "N": "Unpaved",
    },
}

QUALITY_LABELS = {
    "Ex": "Excellent",
    "Gd": "Good",
    "TA": "Typical/average",
    "Fa": "Fair",
    "Po": "Poor",
    "None": "Not present",
}

for column in (
    "ExterQual",
    "KitchenQual",
    "HeatingQC",
    "BsmtQual",
    "FireplaceQu",
):
    CATEGORY_LABELS[column] = QUALITY_LABELS


# ---------------------------------------------------------
# 3. Field guide
# ---------------------------------------------------------

FIELD_GUIDE = [
    ("Neighborhood", "Neighborhood", "Physical neighborhood in Ames, Iowa", "Category"),
    ("Zoning classification", "MSZoning", "General zoning classification", "Category"),
    ("Building type", "BldgType", "Detached home, duplex, townhouse, etc.", "Category"),
    ("House style", "HouseStyle", "Number of stories and dwelling layout", "Category"),
    ("Year built", "YearBuilt", "Original construction year", "Year"),
    ("Year remodeled", "YearRemodAdd", "Most recent remodeling year", "Year"),
    ("Living area", "GrLivArea", "Finished above-ground living area", "Square feet"),
    ("Street frontage", "LotFrontage", "Street length connected to the property", "Linear feet"),
    ("Lot area", "LotArea", "Total land area", "Square feet"),
    ("Basement area", "TotalBsmtSF", "Total basement floor area", "Square feet"),
    ("First-floor area", "1stFlrSF", "First-floor area", "Square feet"),
    ("Second-floor area", "2ndFlrSF", "Second-floor area", "Square feet"),
    ("Overall quality", "OverallQual", "Material and finish quality, 1–10", "Rating"),
    ("Overall condition", "OverallCond", "Current property condition, 1–10", "Rating"),
    ("Exterior quality", "ExterQual", "Exterior material quality", "Quality grade"),
    ("Basement quality", "BsmtQual", "Basement height and quality", "Quality grade"),
    ("Heating quality", "HeatingQC", "Heating system quality", "Quality grade"),
    ("Kitchen quality", "KitchenQual", "Kitchen quality rating", "Quality grade"),
    ("Bedrooms", "BedroomAbvGr", "Bedrooms above basement level", "Count"),
    ("Kitchens", "KitchenAbvGr", "Kitchens above basement level", "Count"),
    ("Rooms above ground", "TotRmsAbvGrd", "Rooms excluding bathrooms", "Count"),
    ("Full bathrooms", "FullBath", "Full bathrooms above ground", "Count"),
    ("Half bathrooms", "HalfBath", "Half bathrooms above ground", "Count"),
    ("Basement full bathrooms", "BsmtFullBath", "Full basement bathrooms", "Count"),
    ("Basement half bathrooms", "BsmtHalfBath", "Half basement bathrooms", "Count"),
    ("Fireplaces", "Fireplaces", "Number of fireplaces", "Count"),
    ("Fireplace quality", "FireplaceQu", "Fireplace quality grade", "Quality grade"),
    ("Garage type", "GarageType", "Garage location and design", "Category"),
    ("Garage finish", "GarageFinish", "Garage interior finish", "Category"),
    ("Garage capacity", "GarageCars", "Number of cars accommodated", "Cars"),
    ("Garage area", "GarageArea", "Garage area", "Square feet"),
    ("Driveway surface", "PavedDrive", "Driveway paving status", "Category"),
    ("Wood deck", "WoodDeckSF", "Wood deck area", "Square feet"),
    ("Open porch", "OpenPorchSF", "Open porch area", "Square feet"),
    ("Enclosed porch", "EnclosedPorch", "Enclosed porch area", "Square feet"),
    ("Screen porch", "ScreenPorch", "Screened porch area", "Square feet"),
    ("Pool area", "PoolArea", "Pool area; zero means no pool", "Square feet"),
    ("Central air", "CentralAir", "Central air conditioning installed", "Yes/No"),
    ("Sale type", "SaleType", "Type of sale transaction", "Category"),
    ("Sale condition", "SaleCondition", "Circumstances of the sale", "Category"),
    ("Sale year", "YrSold", "Year the property was sold", "Year"),
    ("Sale month", "MoSold", "Month the property was sold", "Month"),
]


# ---------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------

def friendly_option(column: str, value: str) -> str:
    """Show a readable label while preserving the original dataset code."""
    code = str(value)
    friendly = CATEGORY_LABELS.get(column, {}).get(code)

    if friendly:
        return f"{friendly} ({code})"

    return code


def option_index(options, value) -> int:
    """Return the index of the default option when it exists."""
    try:
        return list(options).index(value)
    except ValueError:
        return 0


@st.cache_resource
def ensure_model_is_ready():
    """Load and cache the trained prediction pipeline."""
    return load_model()


# ---------------------------------------------------------
# 5. Initialize model and defaults
# ---------------------------------------------------------

try:
    ensure_model_is_ready()
    defaults = default_values()
    stats = reference_price_stats()

except Exception as exc:
    st.error(f"Could not initialize the prediction model: {exc}")
    st.info(
        "Check that the trained model files exist and that "
        "backend/main.py points to the correct paths."
    )
    st.stop()


def select_category(label: str, column: str, help_text: str):
    """Create a category selector using the backend's valid options."""
    options = list(category_options(column))

    if not options:
        raise ValueError(
            f"No category options were returned for column '{column}'."
        )

    default = defaults.get(column, options[0])

    return st.selectbox(
        label,
        options=options,
        index=option_index(options, default),
        format_func=lambda value: friendly_option(column, value),
        help=f"{help_text} Dataset column: {column}.",
        key=f"input_{column}",
    )


def number_input(
    label: str,
    column: str,
    minimum: int,
    maximum: int,
    step: int = 1,
    help_text: str = "",
):
    """Create a number input using the backend's default value."""
    default = int(defaults[column])
    default = max(minimum, min(maximum, default))

    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=step,
        help=f"{help_text} Dataset column: {column}.",
        key=f"input_{column}",
    )


# ---------------------------------------------------------
# 6. Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.title("🏠 Haven")
    st.caption("Ames, Iowa house-price prediction")

    st.info(
        "Use the help text beside each input to understand the field "
        "and its original dataset column."
    )

    st.metric(
        "Training-set median price",
        f"${stats['median']:,.0f}",
    )

    st.metric(
        "Model input columns",
        len(model_features()),
    )

    st.caption(
        "Fields not shown in the form are completed by the backend "
        "using defaults derived from the training data."
    )


# ---------------------------------------------------------
# 7. Page header and tabs
# ---------------------------------------------------------

st.title("🏠 House Price Estimator")

st.markdown(
    "Enter the property's details to estimate its sale price. "
    "This model uses the Ames Housing dataset."
)

estimator_tab, guide_tab, process_tab = st.tabs(
    ["🧮 Estimate", "📖 Field guide", "🔄 How it works"]
)


# ---------------------------------------------------------
# 8. House price estimator
# ---------------------------------------------------------

with estimator_tab:
    st.caption(
        "Enter the main property details. Expand the optional sections "
        "to customize the estimate."
    )

    with st.form("prediction_form"):
        st.subheader("🏠 Main property details")

        location_col, size_col, quality_col = st.columns(3)

        with location_col:
            neighborhood = select_category(
                "Neighborhood",
                "Neighborhood",
                "Physical neighborhood in Ames, Iowa.",
            )

            zoning = select_category(
                "Zoning classification",
                "MSZoning",
                "General zoning classification.",
            )

            building_type = select_category(
                "Building type",
                "BldgType",
                "Detached home, duplex, or townhouse.",
            )

            house_style = select_category(
                "House style",
                "HouseStyle",
                "Number of stories and dwelling layout.",
            )

            year_built = number_input(
                "Year built",
                "YearBuilt",
                1870,
                2010,
                help_text="Original construction year.",
            )

            year_remodeled = number_input(
                "Year remodeled",
                "YearRemodAdd",
                1870,
                2010,
                help_text="Use the build year if never remodeled.",
            )

        with size_col:
            living_area = number_input(
                "Living area above ground (sq ft)",
                "GrLivArea",
                300,
                6000,
                50,
                "Finished living area above the basement.",
            )

            lot_frontage = number_input(
                "Street frontage (linear ft)",
                "LotFrontage",
                20,
                350,
                5,
                "Street length connected to the property.",
            )

            lot_area = number_input(
                "Lot area (sq ft)",
                "LotArea",
                1000,
                220000,
                100,
                "Total land area.",
            )

            basement_area = number_input(
                "Total basement area (sq ft)",
                "TotalBsmtSF",
                0,
                6500,
                50,
                "Enter zero if there is no basement.",
            )

            first_floor = number_input(
                "First-floor area (sq ft)",
                "1stFlrSF",
                300,
                5000,
                50,
                "Area of the first floor.",
            )

            second_floor = number_input(
                "Second-floor area (sq ft)",
                "2ndFlrSF",
                0,
                2500,
                50,
                "Enter zero for a one-story home.",
            )

        with quality_col:
            overall_quality = st.slider(
                "Material and finish quality",
                min_value=1,
                max_value=10,
                value=int(defaults["OverallQual"]),
                help="1 = very poor; 10 = excellent.",
                key="input_OverallQual",
            )

            overall_condition = st.slider(
                "Overall condition",
                min_value=1,
                max_value=10,
                value=int(defaults["OverallCond"]),
                help="1 = very poor; 10 = excellent.",
                key="input_OverallCond",
            )

            exterior_quality = select_category(
                "Exterior material quality",
                "ExterQual",
                "Quality of exterior materials.",
            )

            basement_quality = select_category(
                "Basement quality",
                "BsmtQual",
                "Basement height and quality grade.",
            )

            heating_quality = select_category(
                "Heating system quality",
                "HeatingQC",
                "Heating quality and condition.",
            )

            kitchen_quality = select_category(
                "Kitchen quality",
                "KitchenQual",
                "Overall kitchen quality.",
            )

        # Rooms and bathrooms
        with st.expander("🛏️ Rooms and bathrooms"):
            room_a, room_b, room_c = st.columns(3)

            with room_a:
                bedrooms = number_input(
                    "Bedrooms above basement",
                    "BedroomAbvGr",
                    0, 8,
                )

                kitchens = number_input(
                    "Kitchens above basement",
                    "KitchenAbvGr",
                    0, 3,
                )

                total_rooms = number_input(
                    "Total rooms above ground",
                    "TotRmsAbvGrd",
                    2, 15,
                )

            with room_b:
                full_bath = number_input(
                    "Full bathrooms above ground",
                    "FullBath",
                    0, 4,
                )

                half_bath = number_input(
                    "Half bathrooms above ground",
                    "HalfBath",
                    0, 3,
                )

            with room_c:
                basement_full_bath = number_input(
                    "Basement full bathrooms",
                    "BsmtFullBath",
                    0, 3,
                )

                basement_half_bath = number_input(
                    "Basement half bathrooms",
                    "BsmtHalfBath",
                    0, 2,
                )

        # Garage, fireplace and outdoor areas
        with st.expander("🚗 Garage, fireplace and outdoor areas"):
            st.caption(
                "Enter zero for an absent amenity's count or area. "
                "Choose the appropriate category for the property."
            )

            amenity_a, amenity_b, amenity_c = st.columns(3)

            with amenity_a:
                garage_type = select_category(
                    "Garage type",
                    "GarageType",
                    "Garage location and design.",
                )

                garage_finish = select_category(
                    "Garage finish",
                    "GarageFinish",
                    "Interior finish level.",
                )

                garage_cars = number_input(
                    "Garage capacity (cars)",
                    "GarageCars",
                    0, 5,
                )

                garage_area = number_input(
                    "Garage area (sq ft)",
                    "GarageArea",
                    0, 1500, 25,
                )

                paved_drive = select_category(
                    "Driveway surface",
                    "PavedDrive",
                    "Whether the driveway is paved.",
                )

            with amenity_b:
                fireplaces = number_input(
                    "Number of fireplaces",
                    "Fireplaces",
                    0, 4,
                )

                fireplace_quality = select_category(
                    "Fireplace quality",
                    "FireplaceQu",
                    "Fireplace quality grade.",
                )

                central_air = select_category(
                    "Central air conditioning",
                    "CentralAir",
                    "Whether central air is installed.",
                )

                wood_deck = number_input(
                    "Wood deck area (sq ft)",
                    "WoodDeckSF",
                    0, 900, 10,
                )

            with amenity_c:
                open_porch = number_input(
                    "Open porch area (sq ft)",
                    "OpenPorchSF",
                    0, 750, 10,
                )

                enclosed_porch = number_input(
                    "Enclosed porch area (sq ft)",
                    "EnclosedPorch",
                    0, 600, 10,
                )

                screen_porch = number_input(
                    "Screen porch area (sq ft)",
                    "ScreenPorch",
                    0, 500, 10,
                )

                pool_area = number_input(
                    "Pool area (sq ft)",
                    "PoolArea",
                    0, 800, 10,
                )

        # Sale information
        with st.expander("📄 Sale information"):
            sale_a, sale_b = st.columns(2)

            with sale_a:
                sale_type = select_category(
                    "Sale type",
                    "SaleType",
                    "Type of deed or sale transaction.",
                )

                sale_condition = select_category(
                    "Sale condition",
                    "SaleCondition",
                    "Circumstances of the sale.",
                )

            with sale_b:
                year_sold = number_input(
                    "Sale year",
                    "YrSold",
                    2006,
                    2010,
                )

                month_sold = st.selectbox(
                    "Sale month",
                    options=list(range(1, 13)),
                    index=max(
                        0,
                        min(11, int(defaults["MoSold"]) - 1),
                    ),
                    format_func=lambda month: calendar.month_name[month],
                    help="Month the property was sold. Dataset column: MoSold.",
                    key="input_MoSold",
                )

        submitted = st.form_submit_button(
            "🧮 Estimate property value",
            type="primary",
            use_container_width=True,
        )

    # -----------------------------------------------------
    # 9. Prediction
    # -----------------------------------------------------

    if submitted:
        if year_remodeled < year_built:
            st.error(
                "The remodeling year cannot be earlier than the "
                "original construction year. Please correct the years."
            )
        else:
            payload = {
                "Neighborhood": neighborhood,
                "MSZoning": zoning,
                "BldgType": building_type,
                "HouseStyle": house_style,
                "YearBuilt": year_built,
                "YearRemodAdd": year_remodeled,
                "GrLivArea": living_area,
                "LotFrontage": lot_frontage,
                "LotArea": lot_area,
                "TotalBsmtSF": basement_area,
                "1stFlrSF": first_floor,
                "2ndFlrSF": second_floor,
                "OverallQual": overall_quality,
                "OverallCond": overall_condition,
                "ExterQual": exterior_quality,
                "BsmtQual": basement_quality,
                "HeatingQC": heating_quality,
                "KitchenQual": kitchen_quality,
                "BedroomAbvGr": bedrooms,
                "KitchenAbvGr": kitchens,
                "TotRmsAbvGrd": total_rooms,
                "FullBath": full_bath,
                "HalfBath": half_bath,
                "BsmtFullBath": basement_full_bath,
                "BsmtHalfBath": basement_half_bath,
                "GarageType": garage_type,
                "GarageFinish": garage_finish,
                "GarageCars": garage_cars,
                "GarageArea": garage_area,
                "PavedDrive": paved_drive,
                "Fireplaces": fireplaces,
                "FireplaceQu": fireplace_quality,
                "CentralAir": central_air,
                "WoodDeckSF": wood_deck,
                "OpenPorchSF": open_porch,
                "EnclosedPorch": enclosed_porch,
                "ScreenPorch": screen_porch,
                "PoolArea": pool_area,
                "SaleType": sale_type,
                "SaleCondition": sale_condition,
                "YrSold": year_sold,
                "MoSold": month_sold,
            }

            try:
                with st.spinner("Estimating the property value..."):
                    price = float(predict_price(payload))
                    percentile = float(market_percentile(price))

                st.divider()
                st.subheader("💰 Estimated sale price")

                result_a, result_b = st.columns(2)

                result_a.metric(
                    "Model estimate",
                    f"${price:,.0f}",
                )

                result_b.metric(
                    "Training-data percentile",
                    f"{percentile:.0f}th",
                )

                st.caption(
                    f"The estimate is higher than approximately "
                    f"{percentile:.0f}% of training sale prices. "
                    "This is a model prediction, not a professional appraisal."
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info(
                    "Check backend/main.py, the saved model pipeline, "
                    "and the feature names expected by the model."
                )


# ---------------------------------------------------------
# 10. Field guide
# ---------------------------------------------------------

with guide_tab:
    st.header("📖 Frontend field guide")

    st.markdown(
        "**Frontend label** describes the input shown in the app. "
        "**Dataset column** is the original column sent to the backend."
    )

    guide_df = pd.DataFrame(
        FIELD_GUIDE,
        columns=[
            "Frontend label",
            "Dataset column",
            "Meaning",
            "Value/unit",
        ],
    )

    st.dataframe(
        guide_df,
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("🌐 Category-code legend"):
        st.markdown(
            "Dropdowns display a readable name and the original code. "
            "For example, **Residential low density (RL)** sends `RL` "
            "to `MSZoning`. Quality grades include Ex = excellent, "
            "Gd = good, TA = typical, Fa = fair, and Po = poor."
        )

    with st.expander("📋 All model input columns"):
        visible_columns = {
            row[1] for row in FIELD_GUIDE
        }

        all_fields = []

        for column in model_features():
            if column in ENGINEERED_FEATURES:
                source = "Calculated by backend"
            elif column in visible_columns:
                source = "Editable in frontend"
            else:
                source = "Automatic training-data default"

            all_fields.append({
                "Dataset column": column,
                "How supplied": source,
            })

        st.dataframe(
            pd.DataFrame(all_fields),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Automatic defaults should match the strategy implemented "
            "in backend/main.py."
        )


# ---------------------------------------------------------
# 11. How the model works
# ---------------------------------------------------------

with process_tab:
    st.header("🔄 How the prediction is made")

    st.markdown(
        """
        1. **Enter property details.** The form collects location,
           size, quality, rooms, amenities, and sale information.
        2. **Backend completes the input.** Fields not exposed in the
           form are supplied by the backend.
        3. **Features are engineered.** Derived features are calculated
           using the same logic used during model training.
        4. **The saved pipeline preprocesses the row.** The fitted
           preprocessing steps handle missing values, numerical
           transformations, and categorical encoding.
        5. **The model predicts the price.** The trained pipeline
           returns an estimated sale price in US dollars.
        """
    )

    st.subheader("Calculated feature formulas")

    st.code(
        """TotalBathrooms = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
HouseAge = YrSold - YearBuilt
RemodelAge = YrSold - YearRemodAdd""",
        language="text",
    )

    st.warning(
        "This model was trained on Ames, Iowa home sales from 2006–2010. "
        "Its estimates do not reflect today's market changes, inspection "
        "results, or every unique property characteristic."
    )
