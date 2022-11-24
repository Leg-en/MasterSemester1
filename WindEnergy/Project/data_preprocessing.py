import arcpy
from arcpy import analysis

arcpy.env.workspace = r"C:\workspace\MasterSemester1\WindEnergy\Project\data\study area"
arcpy.env.parallelProcessingFactor = "100%"

processed_data = r"C:\workspace\MasterSemester1\WindEnergy\Project\data\processed_data"

flurstuecke = r"flurstuecke.shp"
potential_areas = r"potential_areas_lpa_400m.shp"
hausumringe = r"hausumringe.shp"

Clipped = processed_data + r"\clipped_data"
analysis.PairwiseClip(flurstuecke, potential_areas, Clipped)

# Erfüllt constraint mit 15m abstand zur flurstücksgrenze
Area_Flurstuecke_bufferd = processed_data + r"\bestehend"
analysis.PairwiseBuffer(Clipped, Area_Flurstuecke_bufferd, "-15 Meters", "NONE", None, "PLANAR", "0 Meters")

# Bufferd_hausumringe = processed_data + r"\bufferd_data"
# analysis.PairwiseBuffer(hausumringe, Bufferd_hausumringe, "50 Meters", "NONE", None, "PLANAR", "0 Meters")
#
# House_Erased_data = processed_data + r"\house_erased"
# analysis.PairwiseErase(Area_Flurstuecke_bufferd, Bufferd_hausumringe, House_Erased_data)


# Ab hier ist die erfüllung des Constraints von mindestens 50m abstand zu einander
bestehend_buffered = processed_data + r"\bestehend_buffered"
analysis.PairwiseBuffer(Area_Flurstuecke_bufferd, bestehend_buffered, "50 Meters", "NONE", None, "PLANAR", "0 Meters")

buffered_intersects = processed_data + r"\buffered_intersects"
analysis.PairwiseIntersect(bestehend_buffered, buffered_intersects, "ALL", None, "INPUT")

Intersects_erased = processed_data + r"\Intersects_erased"
analysis.PairwiseErase(Area_Flurstuecke_bufferd, buffered_intersects, Intersects_erased, None)
