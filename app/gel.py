"""多条已保存酶切记录的凝胶电泳条带只读比较。

把每条记录视为一条泳道，汇总各泳道全部片段的长度，按长度降序做
"分辨率内同条带"聚类：

* 条带以其**最大长度（锚点）**成簇。片段按长度降序逐条处理，只要
  ``(锚点 - 该片段) / 锚点 <= resolution_ratio`` 就并入当前条带，
  否则以该片段为新锚点另开一条条带；
* 该贪心是**完全连接（complete-linkage）**口径：条带内最小片段相对
  锚点都满足分辨率，因此 ``(max-min)/max <= ratio`` 对整条条带恒成立；
  它**不会产生"相邻可合并、首尾超差"的链式归并**——降序处理时，
  若片段连锚点最小（最新）的条带都进不去，则对任何锚点更大的旧
  条带也必然超差，只能另开新条带；
* 等长片段按泳道 id 次序处理，顺序无关，相同请求结果必然一致；
* 整个比较只读取已保存结果、在内存中计算，不读写数据库。
"""

from __future__ import annotations


def build_gel_comparison(lengths_by_lane, resolution_ratio: float) -> dict:
    """比较多条泳道的片段长度，返回按长度降序排列的条带聚类。

    ``lengths_by_lane`` 为 ``[(digest_id, [片段长度...]), ...]``，
    其顺序即结果中泳道的排列顺序（调用方通常已按 digest_id 升序归一化，
    使结果与请求条目的传入顺序无关）。调用方须已完成全部校验
    （泳道数 2~8、id 不重复且记录存在、ratio ∈ [0, 0.2]）。

    返回::

        {
          "lanes": [{"digest_id", "fragment_count", "total_base_pairs"}],
          "bands": [{"min_length", "max_length", "fragment_count",
                     "lanes": [digest_id...], "diagnostic": bool}],
        }

    ``bands`` 按 max_length 降序（锚点严格递减，次序唯一）；
    ``diagnostic`` 标出仅在单条泳道出现的诊断条带。同一泳道的多个
    等长片段可落入同一条带，因此 ``fragment_count`` 为多重集计数。
    """
    lanes = [
        {
            "digest_id": digest_id,
            "fragment_count": len(lengths),
            "total_base_pairs": sum(lengths),
        }
        for digest_id, lengths in lengths_by_lane
    ]

    # 全部片段汇成多重集，按长度降序、等长按泳道 id 升序，保证处理确定
    items = sorted(
        (
            (length, digest_id)
            for digest_id, lengths in lengths_by_lane
            for length in lengths
        ),
        key=lambda item: (-item[0], item[1]),
    )

    clusters: list[dict] = []
    for length, digest_id in items:
        # 只需比较最后一条（锚点最小、最新的）条带：若该片段对它都超差，
        # 则对锚点更大的旧条带同样超差——(anchor-length)/anchor 随 anchor
        # 增大而单调增大——从根上排除链式归并出首尾超差条带的可能。
        if (
            clusters
            and (clusters[-1]["max_length"] - length) / clusters[-1]["max_length"]
            <= resolution_ratio
        ):
            cluster = clusters[-1]
            cluster["min_length"] = length
            cluster["fragment_count"] += 1
            cluster["lane_ids"].add(digest_id)
        else:
            clusters.append(
                {
                    "max_length": length,
                    "min_length": length,
                    "fragment_count": 1,
                    "lane_ids": {digest_id},
                }
            )

    bands = [
        {
            "min_length": cluster["min_length"],
            "max_length": cluster["max_length"],
            "fragment_count": cluster["fragment_count"],
            "lanes": sorted(cluster["lane_ids"]),
            "diagnostic": len(cluster["lane_ids"]) == 1,
        }
        for cluster in clusters
    ]

    return {"lanes": lanes, "bands": bands}
