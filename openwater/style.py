"""
Visual style configuration for OpenWater model categories.

Colors and symbols used for graph visualisation, consistent with
the Periodic Table of OpenWater Models.
"""

# Model category colours (hex) – one per group in the periodic table.
CATEGORY_COLORS = {
    "rainfall-runoff":      "#3B82F6",
    "routing":              "#14B8A6",
    "storage":              "#8B5CF6",
    "sediment-generation":  "#F97316",
    "nutrient-generation":  "#F43F5E",
    "conversion":           "#F59E0B",
    "utility":              "#22C55E",
}

# Map every model name (as registered in the Go catalog) to its category.
MODEL_CATEGORIES = {
    # Rainfall-Runoff
    "RunoffCoefficient":                    "rainfall-runoff",
    "Simhyd":                               "rainfall-runoff",
    "Surm":                                 "rainfall-runoff",
    "GR4J":                                 "rainfall-runoff",
    "Sacramento":                           "rainfall-runoff",

    # Flow Routing
    "Lag":                                  "routing",
    "Muskingum":                            "routing",
    "ConstituentDecay":                     "routing",
    "StorageRouting":                       "routing",
    "LumpedConstituentRouting":             "routing",
    "InstreamFineSediment":                 "routing",
    "InstreamCoarseSediment":               "routing",
    "InstreamDissolvedNutrientDecay":       "routing",
    "InstreamParticulateNutrient":          "routing",

    # Storage / Reservoir
    "StorageTrapAll":                       "storage",
    "StorageDissolvedDecay":                "storage",
    "StorageParticulateTrapping":           "storage",
    "Storage":                              "storage",

    # Sediment Generation
    "PassLoadIfFlow":                       "sediment-generation",
    "BankErosion":                          "sediment-generation",
    "DynamicSednetGully":                   "sediment-generation",
    "DynamicSednetGullyAlt":                "sediment-generation",
    "USLEFineSedimentGeneration":           "sediment-generation",

    # Nutrient Generation
    "SednetDissolvedNutrientGeneration":    "nutrient-generation",
    "EmcDwc":                               "nutrient-generation",
    "SednetParticulateNutrientGeneration":  "nutrient-generation",

    # Conversion & Partition
    "FixedPartition":                       "conversion",
    "VariablePartition":                    "conversion",
    "DeliveryRatio":                        "conversion",
    "ApplyScalingFactor":                   "conversion",
    "DepthToRate":                          "conversion",
    "RatingCurvePartition":                 "conversion",

    # Utility & Climate
    "Input":                                "utility",
    "Sum":                                  "utility",
    "Gate":                                 "utility",
    "ComputeProportion":                    "utility",
    "BaseflowFilter":                       "utility",
    "FixedConcentration":                   "utility",
    "PartitionDemand":                      "utility",
    "DateGenerator":                        "utility",
    "ClimateVariables":                     "utility",
}


def color_for_model(model_name):
    """Return the hex colour string for a given model name."""
    category = MODEL_CATEGORIES.get(model_name)
    if category is None:
        return "#6B7280"  # neutral grey for unknown models
    return CATEGORY_COLORS[category]


def category_for_model(model_name):
    """Return the category key for a given model name, or None."""
    return MODEL_CATEGORIES.get(model_name)
