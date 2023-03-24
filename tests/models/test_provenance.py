import pytest

import sqlalchemy as sa

from models.base import Session
from models.provenance import CodeHash, CodeVersion, Provenance


def test_code_versions():
    cv = CodeVersion(version="test_v0.0.1")
    cv.update()

    assert cv.code_hashes is not None
    assert len(cv.code_hashes) == 1
    assert cv.code_hashes[0] is not None
    assert isinstance(cv.code_hashes[0].hash, str)
    assert len(cv.code_hashes[0].hash) == 40

    try:
        with Session() as session:
            session.add(cv)
            session.commit()
            cv_id = cv.id
            git_hash = cv.code_hashes[0].hash
            assert cv_id is not None

        with Session() as session:
            ch = session.scalars(sa.select(CodeHash).where(CodeHash.hash == git_hash)).first()
            cv = session.scalars(sa.select(CodeVersion).where(CodeVersion.version == 'test_v0.0.1')).first()
            assert cv is not None
            assert cv.id == cv_id
            assert cv.code_hashes[0].id == ch.id

        # add old hash
        old_hash = '696093387df591b9253973253756447079cea61d'
        ch2 = session.scalars(sa.select(CodeHash).where(CodeHash.hash == old_hash)).first()
        if ch2 is None:
            ch2 = CodeHash(old_hash)
        cv.code_hashes.append(ch2)

        with Session() as session:
            session.add(cv)
            session.commit()

            assert len(cv.code_hashes) == 2
            assert cv.code_hashes[0].hash == git_hash
            assert cv.code_hashes[1].hash == old_hash
            assert cv.code_hashes[0].code_version_id == cv.id
            assert cv.code_hashes[1].code_version_id == cv.id

        # check that we can remove commits and have that cascaded
        with Session() as session:
            session.add(cv)  # add it back into the new session
            session.delete(ch2)
            session.commit()
            len(cv.code_hashes) == 1
            assert cv.code_hashes[0].hash == git_hash

            # now check the delete orphan
            cv.code_hashes = []
            session.commit()
            assert len(cv.code_hashes) == 0
            orphan_hash = session.scalars(sa.select(CodeHash).where(CodeHash.hash == git_hash)).first()
            assert orphan_hash is None

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
