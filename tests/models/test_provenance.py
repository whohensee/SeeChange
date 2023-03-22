import pytest

import sqlalchemy as sa

from models.base import Session
from models.provenance import CodeVersion, Provenance


def test_code_versions():
    cv = CodeVersion(version="test_v0.0.1")
    cv.update()

    assert cv.commit_hashes is not None
    assert len(cv.commit_hashes) == 1
    assert cv.commit_hashes[0] is not None
    assert isinstance(cv.commit_hashes[0], str)
    assert len(cv.commit_hashes[0]) == 40

    try:
        with Session() as session:
            session.add(cv)
            session.commit()
            cv_id = cv.id
            assert cv_id is not None

        with Session() as session:
            cv = session.scalars(sa.select(CodeVersion).where(CodeVersion.version == 'test_v0.0.1')).first()
            assert cv is not None
            assert cv.id == cv_id

    finally:
        with Session() as session:
            session.execute(sa.delete(CodeVersion).where(CodeVersion.version == 'test_v0.0.1'))
            session.commit()


def test_provenances(code_version):
    p = Provenance(
        code_version=code_version,
        parameters={"test_key": "test_value1"},
        process="test_process",
        upstream_ids=[],
    )

    with Session() as session:
        # adding the provenance also calculates the hash
        session.add(p)
        session.commit()
        p_id = p.id
        assert p_id is not None
        assert p.unique_hash is not None
        assert isinstance(p.unique_hash, str)
        assert len(p.unique_hash) == 64
        hash = p.unique_hash

    p2 = Provenance(
        code_version=code_version,
        parameters={"test_key": "test_value2"},
        process="test_process",
        upstream_ids=[],
    )
    with Session() as session:
        # adding the provenance also calculates the hash
        session.add(p2)
        session.commit()
        p2_id = p2.id
        assert p2_id is not None
        assert p2.unique_hash is not None
        assert isinstance(p2.unique_hash, str)
        assert len(p2.unique_hash) == 64
        assert p2.unique_hash != hash


def test_upstream_relationship(provenance_base, provenance_extra):
    p = Provenance(
        code_version=provenance_base.code_version,
        parameters={"test_key": "test_value1"},
        process="test_downstream_process",
        upstream_ids=[provenance_base.id],
    )

    with Session() as session:
        session.add(p)
        session.commit()
        p_id = p.id
        assert p_id is not None
        assert p.unique_hash is not None
        assert isinstance(p.unique_hash, str)
        assert len(p.unique_hash) == 64
        hash = p.unique_hash

    p = Provenance(
        code_version=provenance_base.code_version,
        parameters={"test_key": "test_value1"},
        process="test_downstream_process",
        upstream_ids=[provenance_base.id, provenance_extra.id],
    )

    with Session() as session:
        session.add(p)
        session.commit()
        p_id = p.id
        assert p_id is not None
        assert p.unique_hash is not None
        assert isinstance(p.unique_hash, str)
        assert len(p.unique_hash) == 64
        # added a new upstream, so the hash should be different
        assert p.unique_hash != hash

    p = Provenance(
        code_version=provenance_base.code_version,
        parameters={"test_key": "test_value1"},
        process="test_downstream_process",
        upstream_ids=[provenance_base.id, provenance_extra.id, 0],
    )

    with Session() as session:
        with pytest.raises(ValueError) as e:
            session.add(p)
            session.commit()

            assert "refer to non-existent Provenance rows" in str(e.value)
