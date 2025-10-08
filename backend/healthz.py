from fastapi import APIRouter, Response

router = APIRouter()

@router.get("/healthz")
def healthz() -> Response:
    # minimal OK; extend later to check Redis / DB / deps
    return Response(content='{"ok": true}', media_type="application/json")
