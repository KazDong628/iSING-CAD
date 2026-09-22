import copy
import json

from contour_agent.dimension_analysis import analyze_dimensions
from contour_agent.provider import ProviderError


def test_conflicting_provider_cannot_change_source_or_geometry():
    source={"records":[{"text":"R40","box":[[1,2],[3,2],[3,4],[1,4]],"gt":"not-provider-input"}]}
    model={"scale":{"status":"resolved"},"entities":[{"id":"arc-1","radius":40.,
           "radius_binding":{"record_id":"r000","nominal":40.}}],"coordinate_system":{"units":"mm"}}
    before=copy.deepcopy((source,model))
    class Provider:
        def normalize(self, rows):
            assert rows==[{"id":"r000","text":"R40"}]
            return {"status":"succeeded","network_requests":1,"http_success":True,"schema_success":True,
                    "dimensions":[{"id":"r000","kind":"radius","nominal":400.,"upper_deviation":None,"lower_deviation":None}]}
    result=analyze_dimensions(source,model,provider=Provider(),use_api=True)
    assert (source,model)==before
    assert result["counts"]["api_conflicts"]==1
    assert not result["dimensions_verified"] and not result["geometry_updated_by_api"]
    assert result["decisions"][0]["bindings"][0]["fitted_radius"]==40.


def test_dimension_failure_retains_local_evidence_without_secret_error_text():
    class Provider:
        def normalize(self, rows):
            raise ProviderError("timeout","private-error-value",network_requests=1)
    result=analyze_dimensions({"records":[{"text":"135±2"}]},{},provider=Provider(),use_api=True)
    assert result["provider"]["status"]=="failed"
    assert result["provider"]["network_requests"]==1
    assert result["decisions"][0]["local"]["nominal"]==135
    assert "private-error-value" not in json.dumps(result)


def test_request_limit_prioritizes_actual_geometry_bindings():
    source={"records":[{"text":f"R{x+1}"} for x in range(25)]}
    class Provider:
        def normalize(self, rows):
            assert len(rows)==16 and rows[0]["id"]=="r024"
            return {"status":"succeeded","dimensions":[{"id":r["id"],"kind":"radius","nominal":int(r["text"][1:]),
                    "upper_deviation":None,"lower_deviation":None} for r in rows]}
    result=analyze_dimensions(source,{"scale":{"bindings":[{"record_id":"r024"}]}},provider=Provider(),use_api=True)
    assert len(result["decisions"])==25
    assert result["counts"]["api_agreed"]==16
    assert result["counts"]["bound_source_records"]==1


def test_offline_analysis_never_calls_provider():
    class Provider:
        def normalize(self, rows):
            raise AssertionError("Offline analysis attempted API")
    result=analyze_dimensions({"records":[{"text":"R40"}]},{},provider=Provider())
    assert result["provider"]["status"]=="disabled"
    assert result["counts"]["api_selected"]==0
