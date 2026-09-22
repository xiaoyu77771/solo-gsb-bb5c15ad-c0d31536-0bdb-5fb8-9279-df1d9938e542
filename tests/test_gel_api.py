"""凝胶条带比较接口与聚类算法测试。"""

import pytest
from fastapi.testclient import TestClient

from app.gel import build_gel_comparison
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


GC = {"name": "GC-cutter", "pattern": "GC", "cut_top": 1, "cut_bottom": 1}
ECORI = {"name": "EcoRI", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5}


def _create(client, sequence, topology="linear", enzymes=None):
    resp = client.post(
        "/digests",
        json={
            "sequence": sequence,
            "topology": topology,
            "enzymes": enzymes if enzymes not in (None, "") else [GC],
        },
    )
    assert resp.status_code == 201
    return resp.json()


def _compare(client, digest_ids, ratio):
    return client.post(
        "/gel-comparisons",
        json={"digest_ids": digest_ids, "resolution_ratio": ratio},
    )


# ---------- 纯聚类算法 ----------


def test_cluster_merges_within_resolution():
    # 100 与 90 在 ratio=0.1 边界恰好同带；85 相对锚点超差，另开一条
    result = build_gel_comparison([(1, [100]), (2, [90]), (3, [85])], 0.1)
    assert result["bands"] == [
        {"min_length": 90, "max_length": 100, "fragment_count": 2,
         "lanes": [1, 2], "diagnostic": False},
        {"min_length": 85, "max_length": 85, "fragment_count": 1,
         "lanes": [3], "diagnostic": True},
    ]


def test_cluster_forbids_chain_merge():
    # 100,95,90,85 且 ratio=0.1：相邻两两都可合并（95/90 对 95 差 5.3%，
    # 90/85 对 90 差 5.6%），但链式归并会得到首尾差 15% > 10% 的条带。
    # 正确口径：锚点 100 收下 95、90（(100-90)/100 = 0.1 边界），
    # 85 对锚点差 15% 超差 -> 另开，绝不产生 [85,100]。
    result = build_gel_comparison([(1, [100, 95, 90, 85])], 0.1)
    assert [(b["min_length"], b["max_length"]) for b in result["bands"]] == [
        (90, 100), (85, 85)
    ]


def test_cluster_zero_ratio_only_equal_lengths_merge():
    result = build_gel_comparison([(1, [100]), (2, [100]), (3, [99])], 0.0)
    assert [(b["min_length"], b["max_length"], b["diagnostic"]) for b in result["bands"]] == [
        (100, 100, False), (99, 99, True)
    ]


def test_cluster_same_lane_duplicate_lengths_count_as_multiset():
    result = build_gel_comparison([(1, [2, 2]), (2, [2])], 0.0)
    assert result["bands"][0]["fragment_count"] == 3
    assert result["bands"][0]["lanes"] == [1, 2]
    assert result["lanes"] == [
        {"digest_id": 1, "fragment_count": 2, "total_base_pairs": 4},
        {"digest_id": 2, "fragment_count": 1, "total_base_pairs": 2},
    ]


# ---------- API ----------


def test_basic_gel_comparison(client):
    # id=1 "TGCA"      -> [2,2]  (总 4)
    # id=2 "TGCAGCAGCT"-> [2,3,3,2] (总 10)，供其它用例
    # id=3 "TGCAAAGCT" -> [2,5,2] (总 9)
    # id=4 "GCAAAG"    -> [1,5]  (总 6)
    a = _create(client, "TGCA")["id"]
    _create(client, "TGCAGCAGCT")
    c = _create(client, "TGCAAAGCT")["id"]
    d = _create(client, "GCAAAG")["id"]

    resp = _compare(client, [a, c, d], 0.2)
    assert resp.status_code == 200
    data = resp.json()

    # 泳道按 digest_id 升序，与请求传入顺序无关
    assert data["lanes"] == [
        {"digest_id": 1, "fragment_count": 2, "total_base_pairs": 4},
        {"digest_id": 3, "fragment_count": 3, "total_base_pairs": 9},
        {"digest_id": 4, "fragment_count": 2, "total_base_pairs": 6},
    ]

    # 条带按最大长度降序；[5,5] 两泳道共有，[2,2] 含泳道 1 的两个等长片段，
    # [1,1] 仅泳道 4 出现 -> 诊断条带
    assert data["bands"] == [
        {"min_length": 5, "max_length": 5, "fragment_count": 2,
         "lanes": [3, 4], "diagnostic": False},
        {"min_length": 2, "max_length": 2, "fragment_count": 4,
         "lanes": [1, 3], "diagnostic": False},
        {"min_length": 1, "max_length": 1, "fragment_count": 1,
         "lanes": [4], "diagnostic": True},
    ]


def test_result_independent_of_request_order(client):
    a = _create(client, "TGCA")["id"]
    b = _create(client, "TGCAGCAGCT")["id"]
    c = _create(client, "TGCAAAGCT")["id"]

    r1 = _compare(client, [a, b, c], 0.2).json()
    r2 = _compare(client, [c, a, b], 0.2).json()
    assert r1 == r2
    assert [lane["digest_id"] for lane in r1["lanes"]] == [a, b, c]


def test_resolution_boundary_merge_and_split(client):
    # EcoRI 在纯 A 序列上不切：各得到一条全长片段 [100] 与 [80]
    big = _create(client, "A" * 100, enzymes=[ECORI])["id"]
    small = _create(client, "A" * 80, enzymes=[ECORI])["id"]

    # (100-80)/100 = 0.2，边界恰好合并为一条非诊断条带
    merged = _compare(client, [big, small], 0.2)
    assert merged.status_code == 200
    assert merged.json()["bands"] == [
        {"min_length": 80, "max_length": 100, "fragment_count": 2,
         "lanes": [big, small], "diagnostic": False}
    ]

    # ratio=0 时两条全长片段分开，均为诊断条带
    split = _compare(client, [big, small], 0.0).json()
    assert [(b["min_length"], b["max_length"], b["diagnostic"]) for b in split["bands"]] == [
        (100, 100, True), (80, 80, True)
    ]


def test_chain_merge_rejected_via_api(client):
    # 构造 [100]/[95]/[90]/[85] 四条单片段泳道（不切酶），ratio=0.1
    ids = [_create(client, "A" * n, enzymes=[ECORI])["id"] for n in (100, 95, 90, 85)]
    resp = _compare(client, ids, 0.1)
    assert resp.status_code == 200
    ranges = [(b["min_length"], b["max_length"]) for b in resp.json()["bands"]]
    assert ranges == [(90, 100), (85, 85)]


def test_circular_record_fragments_included(client):
    # 环形单一切口开环为 1 条全长片段
    circ = _create(client, "ATTCGGGGGA", topology="circular")["id"]
    lin = _create(client, "A" * 10, enzymes=[ECORI])["id"]
    resp = _compare(client, [circ, lin], 0.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["lanes"] == [
        {"digest_id": circ, "fragment_count": 1, "total_base_pairs": 10},
        {"digest_id": lin, "fragment_count": 1, "total_base_pairs": 10},
    ]
    # 两条泳道都是长度 10 -> 同带且非诊断
    assert data["bands"] == [
        {"min_length": 10, "max_length": 10, "fragment_count": 2,
         "lanes": [circ, lin], "diagnostic": False}
    ]


def test_eight_lanes_allowed_nine_rejected(client):
    ids = [_create(client, "A" * (10 + i), enzymes=[ECORI])["id"] for i in range(8)]
    assert _compare(client, ids, 0.1).status_code == 200
    ids.append(_create(client, "A" * 50, enzymes=[ECORI])["id"])
    assert _compare(client, ids, 0.1).status_code == 422


def test_too_few_ids_returns_422(client):
    one = _create(client, "TGCA")["id"]
    assert _compare(client, [one], 0.1).status_code == 422
    assert _compare(client, [], 0.1).status_code == 422


def test_duplicate_ids_return_422(client):
    a = _create(client, "TGCA")["id"]
    b = _create(client, "GCAAAG")["id"]
    resp = _compare(client, [a, b, a], 0.1)
    assert resp.status_code == 422


def test_ratio_out_of_range_returns_422(client):
    a = _create(client, "TGCA")["id"]
    b = _create(client, "GCAAAG")["id"]
    for bad in (-0.01, 0.21):
        assert _compare(client, [a, b], bad).status_code == 422


def test_nan_or_infinite_ratio_returns_422(client):
    a = _create(client, "TGCA")["id"]
    b = _create(client, "GCAAAG")["id"]
    for raw in ('"NaN"', '"Infinity"', '"-Infinity"'):
        resp = client.post(
            "/gel-comparisons",
            content=f'{{"digest_ids": [{a}, {b}], "resolution_ratio": {raw}}}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, raw


def test_missing_record_returns_404(client):
    a = _create(client, "TGCA")["id"]
    resp = _compare(client, [a, 999], 0.1)
    assert resp.status_code == 404
    # 全部不存在同样 404
    assert _compare(client, [998, 999], 0.1).status_code == 404


def test_negative_id_treated_as_missing_404(client):
    a = _create(client, "TGCA")["id"]
    resp = _compare(client, [a, -1], 0.1)
    assert resp.status_code == 404


def test_comparison_is_read_only(client):
    a = _create(client, "TGCA")["id"]
    b = _create(client, "GCAAAG")["id"]
    assert _compare(client, [a, b], 0.1).status_code == 200
    # 原记录保持不变，也没有新增任何提交
    listing = client.get("/digests").json()
    assert [item["id"] for item in listing] == [1, 2]
