def assemble_headway_constraints(
    trip_data: dict,
    trip_data_enriched: dict,
    routes: dict,
    nodesDf: pd.DataFrame,
    chains: dict,
    headway_dict: dict,
) -> tuple[list[dict], list]:

    def time_to_sec(t):
        return t.hour * 3600 + t.minute * 60 + t.second

    def event_time_sec(train_id, seq, event):
        t = trip_data_enriched[train_id][seq][2 if event == "dep" else 1]
        return time_to_sec(t)

    def category(speed, direction):
        return f"{speed} {direction}"

    chain_keys_in_headways = {k[0] for k in headway_dict}

    constraints = []
    skipped = []

    for chain_key in chain_keys_in_headways:

        resolved = chains_key_lookup(chains, chain_key)
        if resolved is None:
            skipped.append(("unresolved chain", chain_key))
            continue

        boundary_a, boundary_b = resolved

        occupants = []

        # --------------------------------------------------
        # Find trains using this chain
        # --------------------------------------------------
        for train_id, route in routes.items():

            if boundary_a not in route or boundary_b not in route:
                continue

            idx_a = route.index(boundary_a)
            idx_b = route.index(boundary_b)

            if idx_a < idx_b:
                entry_node, exit_node = boundary_a, boundary_b
            else:
                entry_node, exit_node = boundary_b, boundary_a

            entry_evt = chain_boundary_event(
                train_id,
                route,
                trip_data_enriched,
                entry_node,
                "entry",
            )

            exit_evt = chain_boundary_event(
                train_id,
                route,
                trip_data_enriched,
                exit_node,
                "exit",
            )

            if entry_evt is None or exit_evt is None:
                skipped.append(
                    ("no EAN event on chain", chain_key, train_id)
                )
                continue

            entry_seq, entry_event = entry_evt
            exit_seq, exit_event = exit_evt

            speed = train_speed_category(
                train_id,
                route,
                trip_data,
                nodesDf,
            )

            direction = train_direction(route, nodesDf)

            occupants.append(
                {
                    "train": train_id,
                    "speed": speed,
                    "direction": direction,

                    "entry_seq": entry_seq,
                    "entry_event": entry_event,
                    "entry_time": event_time_sec(
                        train_id,
                        entry_seq,
                        entry_event,
                    ),

                    "exit_seq": exit_seq,
                    "exit_event": exit_event,
                    "exit_time": event_time_sec(
                        train_id,
                        exit_seq,
                        exit_event,
                    ),
                }
            )

        # --------------------------------------------------
        # Consecutive trains only
        # --------------------------------------------------
        occupants.sort(key=lambda x: x["entry_time"])

        for first, second in zip(occupants, occupants[1:]):

            cat1 = category(
                first["speed"],
                first["direction"],
            )

            cat2 = category(
                second["speed"],
                second["direction"],
            )

            # ----------------------------------------------
            # Same direction:
            # entry departure -> next entry departure
            # ----------------------------------------------
            if first["direction"] == second["direction"]:

                seq_i = first["entry_seq"]
                event_i = first["entry_event"]

                seq_j = second["entry_seq"]
                event_j = second["entry_event"]

            # ----------------------------------------------
            # Opposite direction:
            # exit -> next entry
            # ----------------------------------------------
            else:
                if first["exit_time"] <= second["entry_time"]:

                    seq_i = first["exit_seq"]
                    event_i = first["exit_event"]

                    seq_j = second["entry_seq"]
                    event_j = second["entry_event"]

                    cat1 = category(
                        first["speed"],
                        first["direction"],
                    )

                    cat2 = category(
                        second["speed"],
                        second["direction"],
                    )

                elif second["exit_time"] <= first["entry_time"]:

                    seq_i = second["exit_seq"]
                    event_i = second["exit_event"]

                    seq_j = first["entry_seq"]
                    event_j = first["entry_event"]

                    cat1 = category(
                        second["speed"],
                        second["direction"],
                    )

                    cat2 = category(
                        first["speed"],
                        first["direction"],
                    )

                else:
                    skipped.append(
                        (
                            "overlapping opposite trains",
                            chain_key,
                            first["train"],
                            second["train"],
                        )
                    )
                    continue

            key = (chain_key, cat1, cat2)

            if key not in headway_dict:
                skipped.append(
                    (
                        "category pair missing",
                        chain_key,
                        cat1,
                        cat2,
                    )
                )
                continue

            constraints.append(
                {
                    "train_i": first["train"],
                    "seq_i": seq_i,
                    "event_i": event_i,

                    "train_j": second["train"],
                    "seq_j": seq_j,
                    "event_j": event_j,

                    "min_headway": headway_dict[key],
                    "resource": chain_key,
                }
            )

    if skipped:
        print(
            f"[assemble_headway_constraints] {len(skipped)} entries skipped "
            "-- inspect `skipped` for details."
        )

    return constraints, skipped
