from fastapi import APIRouter

from services import lulc_transition_service as transition_svc

router = APIRouter(prefix="/change", tags=["change"])


@router.get("/aois")
def change_aois():
    """Alias under /change for AOI lists (canonical endpoint is GET /aois)."""
    return transition_svc.list_aois()
