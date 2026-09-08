from shadow_lab.mage_w4_search.data import freeze_pilot_manifest


def test_source_variants_stay_together_and_reserve_is_sealed():
    rows = [
        {"sample_id": "a", "prompt": "A blue ceramic cup", "source": "old", "source_id": "1"},
        {"sample_id": "b", "prompt": "A green ceramic cup", "source": "old", "source_id": "1"},
        {"sample_id": "c", "prompt": "A forest river", "source": "old", "source_id": "2"},
    ]
    manifest, audit = freeze_pilot_manifest(rows)
    assert manifest.validate() == ()
    a, b = manifest.records[:2]
    assert a.source_group == b.source_group
    assert a.development_uses == b.development_uses
    sealed = [r for r in manifest.records if r.pool != "development"]
    assert len(sealed) == 64
    assert all(not r.development_uses and not r.retired for r in sealed)
    assert audit["sealed_images_generated"] == 0
    assert audit["formal_sample_sizes_frozen"] is False
