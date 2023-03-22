import pytest
import uuid

import sqlalchemy as sa

from models.base import Session
from models.provenance import CodeVersion, Provenance


@pytest.fixture(scope="session", autouse=True)
def code_version():
    cv = CodeVersion(version="test_v1.0.0")
    cv.update()

    yield cv

    with Session() as session:
        session.execute(sa.delete(CodeVersion).where(CodeVersion.version == 'test_v1.0.0'))
        session.commit()


@pytest.fixture
def provenance_base(code_version):
    p = Provenance(
        code_version=code_version,
        parameters={"test_key": uuid.uuid4().hex},
        process="test_base_process",
        upstream_ids=[],
    )

    with Session() as session:
        session.add(p)
        session.commit()
        pid = p.id

    yield p

    with Session() as session:
        session.execute(sa.delete(Provenance).where(Provenance.id == pid))
        session.commit()


@pytest.fixture
def provenance_extra(code_version, provenance_base):
    p = Provenance(
        code_version=code_version,
        parameters={"test_key": uuid.uuid4().hex},
        process="test_base_process",
        upstream_ids=[provenance_base.id],
    )

    with Session() as session:
        session.add(p)
        session.commit()
        pid = p.id

    yield p

    with Session() as session:
        session.execute(sa.delete(Provenance).where(Provenance.id == pid))
        session.commit()

