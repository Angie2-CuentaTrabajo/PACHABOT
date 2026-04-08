from app.models.domain import QueryIntent
from app.services.query_router import QueryRouter


def test_router_detects_payments() -> None:
    router = QueryRouter()
    routed = router.route("Cuanto se paga de SISA para comercio ambulatorio")
    assert routed.in_domain is True
    assert routed.intent == QueryIntent.PAGOS_SISA


def test_router_rejects_out_of_scope_question() -> None:
    router = QueryRouter()
    routed = router.route("Cual es el horario de atencion del impuesto predial")
    assert routed.in_domain is False
    assert routed.intent == QueryIntent.OUT_OF_SCOPE


def test_router_accepts_generic_article_question() -> None:
    router = QueryRouter()
    routed = router.route("Dime que dice el articulo 7")
    assert routed.in_domain is True
    assert routed.intent == QueryIntent.GENERAL


def test_router_accepts_natural_municipal_case_without_exact_keywords() -> None:
    router = QueryRouter()
    routed = router.route("Puedo poner una carretilla para vender en la calle")
    assert routed.in_domain is True
