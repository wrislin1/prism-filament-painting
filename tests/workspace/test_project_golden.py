"""Invented saved states. Each ID, colour, lot, grid and observation is synthetic."""


from uuid import UUID


def spools():
    return [
        dict(brand="Kite", name="Flint", hex="#26314b", td=2.0, spool_id="coil-a"),
        dict(brand="Kite", name="Brume", hex="#d2a869", td=3.0, spool_id="coil-b",
             opacity_level="measured", td_source="wedge", color_level="vendor"),
        dict(brand="Kite", name="Lichen", hex="#51917a", td=1.25, opacity_level="estimated"),
        dict(brand="Kite", name="Silt", hex="#d8dce3", td=4., color_level="measured"),
    ]


def image():
    return dict(project_id=str(UUID(int=7, version=4)), upload_token=None,
                filename="invented-ramp.png", content_type="image/png")


def classic():
    return dict(version=1, image=image(), palette=dict(filaments=spools(), score=0.5, notes="Ramp"),
                td_scale=0.125, layer=dict(base_layers=2, total_layers=14, layer_height_mm=0.125),
                swap_schedule=[4, 8, 11], mesh=dict(max_dim_mm=14., xy_pixel_mm=1.,
                max_triangles=4096, flatten_transparent=True, orientation="bright"),
                luminance=dict(black_point=0., white_point=1., gamma=1.),
                print_options=dict(smoothing=0.25, dither=True), stored_at=12345.)


def flat():
    palette = spools()
    regions = [dict(name="Left", slots=[2], mask=["10", "10"]),
               dict(name="Lower", slots=[3], mask=["00", "11"])]
    options = dict(profile="u1", plan="full", background_slots=[1], regions=regions, dither_scope=2,
                   slice_report=dict(project_key="12"*32, slicer="Example", minutes=3.25,
                                     flushed_g=0.5, tower_g=0.25, passes=[[1], [2, 3]]))
    recipes = [dict(slots=[s]*4, rgb=rgb, source="formula")
               for s, rgb in enumerate([[38, 49, 75], [204, 159, 102], [81, 145, 122]], 1)]
    run = dict(schema_version=1, profile="u1", selected_plan="full",
               geometry=dict(rows=2, columns=2, cell_mm=1., picture_layers=4,
                             picture_layer_height_mm=0.125, backing_layer_heights_mm=[0.2, 0.12]),
               recipe_table=recipes, cell_recipe_ids=[[1, 0], [2, 2]],
               regions=dict(palettes=[[1], [2], [3]], cell_regions=[[1, 0], [2, 2]]),
               dithering=dict(scope=2), plans={"full": dict(allowed_picture_slots=[[1, 2, 3, 4]]*4)},
               model=dict(version="synthetic-v1", td_scale=0.125),
               slots=[dict(slot=i, **spool) for i, spool in enumerate(palette, 1)])
    return dict(version=1, project_type="flat", image=image(), palette=dict(filaments=palette),
                options=options, flat=dict(mode="flat", run_id="quilt_0000000000000007", run=run,
                                           cost=dict(basis="invented example", minutes=3.25)))


def test_version_two_documents_match_complete_goldens():
    from copy import deepcopy

    from prism_filament_painting.palette.project_migration import adapt_project

    for make in (classic, flat):
        original = make()

        def check_original(doc):
            # The adapter must pass the whole legacy document to its validator.
            assert doc == original

        expected = deepcopy(original)
        expected["version"] = 2
        expected.pop("td_scale", None)
        expected["front_lit_scale"] = 0.125 if make == flat else 0.1
        if make == classic:
            expected["v1_declared_scale"] = 0.125
        groups = [expected["palette"]["filaments"]]
        if make == flat:
            run = expected["flat"]["run"]
            groups.append(run["slots"])
            del run["model"]["td_scale"]
            run["model"]["front_lit_scale"] = 0.125
        for group in groups:
            for spool, value, source, color in zip(
                group, (2., 3., 1.25, 4.), ("unknown", "wedge", "estimated", "unknown"),
                ("unknown", "vendor", "unknown", "measured"),
                strict=True,
            ):
                del spool["td"]
                spool.update(input_distance_mm=value, input_distance_source=source,
                             color_source=color, hiding_distance_mm=value * expected["front_lit_scale"])
        assert adapt_project(original, validate_v1=check_original) == expected
        assert original == make()


def test_unrecognised_provenance_is_preserved_as_evidence():
    from prism_filament_painting.palette.project_migration import adapt_project

    state = classic()
    state["palette"]["filaments"][0]["td_source"] = "archive-card"
    result = adapt_project(state, validate_v1=classic_geometry)
    spool = result["palette"]["filaments"][0]
    assert spool["input_distance_source"] == "unknown"
    assert spool["td_source"] == "archive-card"


def classic_geometry(doc):
    assert doc["layer"] == classic()["layer"]
