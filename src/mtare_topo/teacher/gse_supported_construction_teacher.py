"""Construction labels with an explicit, limited axis-support interval proxy.

Teacher ONLY. No model, file access, renderer, region mesh, or radius tuning.
Interval coverage can span unobserved gaps. It is not a proof of free space,
physical connectivity, a complete detector annotation, or robot clearance.
"""
from itertools import combinations

import numpy as np

from mtare_topo.data.cano_sensor_smoke import NEAR_RANGE_M, MAX_RANGE_M

from mtare_topo.data.primitive_frame_support import (
    summarize_primitive_frame_support, visible_primitive_targets_from_frame_support,
    _current_sensor_transform,
)
from mtare_topo.data.primitive_relation_targets import primitive_relation_targets_from_frame_support
from mtare_topo.data.primitive_relation_storage import pack_primitive_relation_targets
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField


def construction_source_conflicts(construction, primitive_ids, observed_source_sets):
    """Multi-source nonincident provenance is ambiguity, NOT overlap proof."""
    lookup = {name: i for i, name in enumerate(primitive_ids)}
    incident = set()
    for group in construction.compositions:
        members = sorted({lookup[x.primitive_id] for x in group.member_endpoints})
        incident.update(combinations(members, 2))
    conflicts = set()
    for values in observed_source_sets:
        if (tuple(sorted(set(values))) != tuple(values)
                or any(type(i) not in (int, np.int64, np.int32) or not 0 <= i < len(primitive_ids) for i in values)):
            raise ValueError("invalid observed provenance set")
        for pair in combinations(values, 2):
            if pair not in incident:
                conflicts.add(pair)
    return tuple(sorted(conflicts))


def window_supported_targets(*, construction, field, frame_supports,
                             current_sensor_xyz_m, current_yaw_deg,
                             nonincident_source_pairs=()):
    if len(frame_supports) != 5 or field.spacing_m != .025:
        raise ValueError("exact five frames and native 0.025m axis sampling required")
    count = len(field.primitive_ids)
    if any(len(s.support_ray_count) != count for s in frame_supports):
        raise ValueError("support population differs from native field")
    lookup = {name: i for i, name in enumerate(field.primitive_ids)}
    # Validate the full construction before visibility filtering: an invisible
    # malformed composition must not escape the same ownership contract.
    node_ids, endpoint_owners = set(), set()
    for group in construction.compositions:
        if type(group.node_id) is not str or not group.node_id or group.node_id in node_ids:
            raise ValueError("construction node IDs must be unique nonempty strings")
        node_ids.add(group.node_id)
        if (type(group.degree) is not int or group.degree < 1
                or group.degree != len(group.member_endpoints)):
            raise ValueError("construction degree must equal nonempty endpoint incidence")
        for member in group.member_endpoints:
            if member.node_id != group.node_id:
                raise ValueError("construction member node ownership differs from composition")
            key = (member.primitive_id, member.endpoint_index)
            if (member.primitive_id not in lookup or type(member.endpoint_index) is not int
                    or member.endpoint_index not in (0, 1) or key in endpoint_owners):
                raise ValueError("construction endpoint ownership ambiguous or unknown")
            endpoint_owners.add(key)
    total = np.sum([s.support_ray_count for s in frame_supports], axis=0)
    visible = np.flatnonzero(total > 0)
    if len(visible) > 32:
        raise ValueError("visible fragment count exceeds unchanged 32-slot contract")
    slot_for = {int(i): slot for slot, i in enumerate(visible)}
    conflict_primitives = set()
    for pair in nonincident_source_pairs:
        if len(pair) != 2 or any(not 0 <= i < count for i in pair):
            raise ValueError("invalid source conflict")
        conflict_primitives.update(pair)
    all_groups = []
    for group in construction.compositions:
        if group.anchor_xyz_m is None:
            raise ValueError("construction anchor missing; no cropped-end fallback")
        if not any(lookup[m.primitive_id] in slot_for for m in group.member_endpoints):
            continue
        members = []
        for member in group.member_endpoints:
            index = lookup[member.primitive_id]
            operand = field.operands[index]
            distances, indices = operand.tree.query(np.asarray(group.anchor_xyz_m), k=2, workers=1)
            # Source quantization scale is unchanged. Equidistant projections
            # get UNKNOWN rather than a slot/index tie-breaking supervision.
            scale = max(1., float(np.linalg.norm(group.anchor_xyz_m)), float(distances[1]))
            tied = abs(float(distances[1] - distances[0])) <= 32 * np.finfo(np.float64).eps * scale
            projected = int(indices[0])
            supports = [s for s in frame_supports if s.support_ray_count[index] > 0]
            low = min((int(s.minimum_sample_index[index]) for s in supports), default=-1)
            high = max((int(s.maximum_sample_index[index]) for s in supports), default=-1)
            if supports and (low < 0 or high >= len(operand.points) or high < low):
                raise ValueError("support interval out of native axis bounds")
            reasons = []
            if not supports: reasons.append("NO_SOURCE_RETURN")
            elif not low <= projected <= high: reasons.append("ANCHOR_OUTSIDE_SUPPORT_INTERVAL")
            if tied: reasons.append("ANCHOR_PROJECTION_NUMERIC_TIE")
            if index in conflict_primitives: reasons.append("NONINCIDENT_SOURCE_AMBIGUITY")
            members.append({"primitive_id_teacher_only": member.primitive_id,
                "endpoint_index_teacher_only": int(member.endpoint_index),
                "teacher_direction_token": (2 * slot_for[index] + member.endpoint_index) if index in slot_for else None,
                "projected_native_axis_index": projected, "support_min_index": low, "support_max_index": high,
                "support_ray_count": int(total[index]), "supported": not reasons, "unknown_reasons": reasons})
        any_supported = any(m["supported"] for m in members)
        all_supported = all(m["supported"] for m in members)
        if not members: raise ValueError("empty construction composition")
        event = "corridor" if len(members) == 2 else "junction" if len(members) >= 3 else "terminal"
        # No cap face evidence is implemented/claimed by this pilot. It must
        # not quietly promote degree=1 or absent aperture crossing to terminal.
        event_valid = all_supported and event != "terminal"
        all_groups.append({"construction_node_id_teacher_only": str(group.node_id),
            "center_current_sensor_m": _current_sensor_transform(np.asarray(group.anchor_xyz_m), current_sensor_xyz_m, current_yaw_deg).tolist(),
            "center_valid": any_supported, "event_target": event if event_valid else None,
            "event_valid": event_valid, "construction_degree_teacher_only": len(members),
            "event_unknown_reason": None if event_valid else "TERMINAL_CAP_RETURN_NOT_VERIFIED" if event == "terminal" else "INCIDENT_SUPPORT_INCOMPLETE_OR_AMBIGUOUS",
            "members": members, "interval_proxy_not_continuous_visibility_proof": True})
    positive_owners = {}
    for group in all_groups:
        for member in group["members"]:
            if member["supported"]:
                token = member["teacher_direction_token"]
                if token in positive_owners: raise ValueError("supported direction belongs to multiple instances")
                positive_owners[token] = group["construction_node_id_teacher_only"]
    for group in all_groups:
        labels, valid = [0] * 64, [False] * 64
        if group["center_valid"]:
            for token, node in positive_owners.items():
                labels[token], valid[token] = int(node == group["construction_node_id_teacher_only"]), True
        group["directional_member_target"], group["directional_member_valid"] = labels, valid
    return {"regions": all_groups, "visible_fragments": len(visible),
        "nonincident_source_pairs_teacher_only": [list(p) for p in nonincident_source_pairs],
        "observation_label_complete": False, "terminal_cap_evidence_implemented": False,
        "physical_connectivity_certified": False}


def build_task_targets(*, teacher, sensor, sensor_frame_rows, construction, codebook):
    from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
    from mtare_topo.data.primitive_relation_sequences import causal_relative_odometry
    rows_array = np.asarray(teacher["frame_row"])
    if (rows_array.ndim != 2 or rows_array.shape[1] != 5 or rows_array.dtype.kind not in "iu"
            or (rows_array < 0).any() or not (rows_array[:, 1:] > rows_array[:, :-1]).all()):
        raise ValueError("noncausal/noninteger selected frame rows")
    sequence_indices = np.asarray(teacher["source_global_sequence_index"])
    if (sequence_indices.shape != rows_array.shape[:1] or sequence_indices.dtype.kind not in "iu"
            or (sequence_indices < 0).any() or len(np.unique(sequence_indices)) != len(sequence_indices)):
        raise ValueError("invalid selected sequence indices")
    frames = len(sensor_frame_rows)
    for key, dtype, shape in (("range_m", np.float32, (frames,16,720)), ("valid_mask",np.uint8,(frames,16,720)),
                              ("primitive_membership_code",np.uint16,(frames,16,720))):
        if sensor[key].shape != shape or sensor[key].dtype != dtype or not np.isfinite(sensor[key]).all():
            raise ValueError(f"sensor shape/dtype/nonfinite: {key}")
    for key, shape in (("sensor_xyz_m",(frames,3)),("yaw_deg",(frames,))):
        if sensor[key].shape != shape or sensor[key].dtype not in (np.float32,np.float64) or not np.isfinite(sensor[key]).all():
            raise ValueError(f"invalid teacher-only sensor pose: {key}")
    if (not np.isin(sensor["valid_mask"], (0,1)).all() or (sensor["range_m"] < NEAR_RANGE_M).any()
            or (sensor["range_m"] > MAX_RANGE_M).any()):
        raise ValueError("sensor valid/range contract drift")
    # The legacy loader reconstructs degree from members and does not retain
    # the JSON degree field. Check its declared value before that information
    # would be discarded; no silent repair of a malformed construction record.
    base = construction.get("base_construction")
    if not isinstance(base, dict) or not isinstance(base.get("composition_operations"), list):
        raise ValueError("construction composition records missing")
    for record in base["composition_operations"]:
        if (not isinstance(record, dict) or not isinstance(record.get("member_endpoints"), list)
                or type(record.get("degree")) is not int or record["degree"] < 1
                or record["degree"] != len(record["member_endpoints"])):
            raise ValueError("declared JSON degree differs from endpoint incidence")
    graph, primitives = load_p1a_realized_construction(construction)
    field = SweptSuperellipseProvenanceField(primitives, spacing_m=.025)
    if tuple(codebook["primitive_ids"]) != field.primitive_ids:
        raise ValueError("codebook/native field primitive ordering drift")
    raw_sources = codebook["source_sets"]
    if any(any(type(i) is not int or not 0 <= i < len(field.primitive_ids) for i in values)
           or list(values) != sorted(set(values)) for values in raw_sources):
        raise ValueError("source sets require exact sorted unique integer indices")
    sources = tuple(tuple(i for i in values) for values in raw_sources)
    if not sources or sources[0] != (): raise ValueError("source zero code must be no return")
    if any(not source for source in sources[1:]) or len(set(sources)) != len(sources):
        raise ValueError("duplicate/empty nonzero source codes")
    sensor_frame_rows = np.asarray(sensor_frame_rows)
    if (sensor_frame_rows.dtype.kind not in "iu" or len(np.unique(sensor_frame_rows)) != len(sensor_frame_rows)
            or not np.array_equal(sensor_frame_rows, np.unique(teacher["frame_row"]))):
        raise ValueError("exact selected unique frame population drift")
    frame_lookup = {int(row): i for i, row in enumerate(sensor_frame_rows)}
    supports = []
    for index in range(len(sensor_frame_rows)):
        if not np.array_equal(sensor["valid_mask"][index] > 0, sensor["primitive_membership_code"][index] > 0):
            raise ValueError("valid mask/source-code evidence drift")
        supports.append(summarize_primitive_frame_support(range_m=sensor["range_m"][index],
            primitive_membership_code=sensor["primitive_membership_code"][index], source_sets=sources,
            field=field, sensor_xyz_m=sensor["sensor_xyz_m"][index], yaw_deg=float(sensor["yaw_deg"][index])))
    rows = []
    fields = {"primitive_index": "primitive_index", "primitive_mask": "mask",
        "axis_control_current_sensor_m": "axis_control_current_sensor_m", "endpoint_half_axes_m": "endpoint_half_axes_m",
        "endpoint_shape_exponent": "endpoint_shape_exponent", "support_ray_count": "support_ray_count", "temporal_visibility": "temporal_visibility"}
    for row, frame_rows in enumerate(teacher["frame_row"]):
        if len(frame_rows) != 5 or not np.all(frame_rows[1:] > frame_rows[:-1]): raise ValueError("noncausal observation")
        positions = [frame_lookup[int(x)] for x in frame_rows]
        current = positions[-1]
        frame_supports = tuple(supports[x] for x in positions)
        visible = visible_primitive_targets_from_frame_support(field=field, frame_supports=frame_supports,
            current_sensor_xyz_m=sensor["sensor_xyz_m"][current], current_yaw_deg=float(sensor["yaw_deg"][current]), maximum_slots=32)
        for key, attr in fields.items():
            if not np.array_equal(teacher[key][row], getattr(visible, attr)):
                raise ValueError(f"old teacher raw reconstruction differs: {key}, row={row}")
        relation = primitive_relation_targets_from_frame_support(construction=graph, primitive_ids=field.primitive_ids,
            frame_supports=frame_supports, maximum_slots=32)
        packed = pack_primitive_relation_targets(relation)
        for name in ("endpoint_neighbor", "disconnected_overlap_packed"):
            if not np.array_equal(teacher[name][row], getattr(packed, name)):
                raise ValueError(f"old relation reconstruction differs: {name}")
        odometry = causal_relative_odometry(sensor["sensor_xyz_m"][positions], sensor["yaw_deg"][positions])
        for name, values in (("relative_translation_current_sensor_m", odometry.translation_current_sensor_m),
                             ("relative_yaw_current_sensor_deg", odometry.yaw_current_sensor_deg)):
            if not np.array_equal(teacher[name][row], values): raise ValueError(f"odometry reconstruction differs: {name}")
        observed_codes = np.unique(sensor["primitive_membership_code"][positions])
        conflict = construction_source_conflicts(graph, field.primitive_ids, [sources[int(c)] for c in observed_codes])
        output = window_supported_targets(construction=graph, field=field, frame_supports=frame_supports,
            current_sensor_xyz_m=sensor["sensor_xyz_m"][current], current_yaw_deg=float(sensor["yaw_deg"][current]),
            nonincident_source_pairs=conflict)
        rows.append({"row_in_selected_task": row, "source_global_sequence_index": int(teacher["source_global_sequence_index"][row]),
            "frame_rows": [int(x) for x in frame_rows], **output})
    events = {event: sum(g["event_target"] == event for r in rows for g in r["regions"]) for event in ("corridor", "junction", "terminal")}
    return {"rows": rows, "counts": {"observations": len(rows), "unique_sensor_frames": len(sensor_frame_rows),
        "visible_fragments": sum(r["visible_fragments"] for r in rows),
        "candidate_region_instances": sum(len(r["regions"]) for r in rows),
        "center_labels": sum(g["center_valid"] for r in rows for g in r["regions"]), "events": events,
        "member_positive": sum(sum(v and t == 1 for v, t in zip(g["directional_member_valid"], g["directional_member_target"])) for r in rows for g in r["regions"]),
        "member_negative": sum(sum(v and t == 0 for v, t in zip(g["directional_member_valid"], g["directional_member_target"])) for r in rows for g in r["regions"]),
        "observations_with_nonincident_source_ambiguity": sum(bool(r["nonincident_source_pairs_teacher_only"]) for r in rows)},
        "old_teacher_all_fields_reconstructed_exactly": True, "capacity_ready": False,
        "three_class_capacity_reason": "TERMINAL_CAP_EVIDENCE_NOT_IMPLEMENTED_NO_THREE_CLASS_PASS"}
