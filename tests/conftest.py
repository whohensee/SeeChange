import os
import pytest
import uuid

import numpy as np

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import CodeVersion, Provenance
from models.exposure import Exposure


def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


@pytest.fixture(scope="session", autouse=True)
def code_version():
    cv = CodeVersion(version="test_v1.0.0")
    cv.update()

    yield cv

    with SmartSession() as session:
        session.execute(sa.delete(CodeVersion).where(CodeVersion.version == 'test_v1.0.0'))
        session.commit()


@pytest.fixture
def provenance_base(code_version):
    p = Provenance(
        process="test_base_process",
        code_version=code_version,
        parameters={"test_key": uuid.uuid4().hex},
        upstreams=[],
    )

    with SmartSession() as session:
        session.add(p)
        session.commit()
        pid = p.id

    yield p

    with SmartSession() as session:
        session.execute(sa.delete(Provenance).where(Provenance.id == pid))
        session.commit()


@pytest.fixture
def provenance_extra(code_version, provenance_base):
    p = Provenance(
        process="test_base_process",
        code_version=code_version,
        parameters={"test_key": uuid.uuid4().hex},
        upstreams=[provenance_base],
    )

    with SmartSession() as session:
        session.add(p)
        session.commit()
        pid = p.id

    yield p

    with SmartSession() as session:
        session.execute(sa.delete(Provenance).where(Provenance.id == pid))
        session.commit()


@pytest.fixture
def exposure():
    e = Exposure(
        f"Demo_test_{rnd_str(5)}.fits",
        section_id=0,
        exp_time=30,
        mjd=58392.0,
        filter="g",
        ra=123,
        dec=-23,
        project='foo',
        target='bar',
        nofile=True,
    )
    fullname = None
    try:  # make sure to remove file at the end
        fullname = e.get_fullpath()
        open(fullname, 'a').close()
        e.nofile = False

        yield e

    finally:
        with SmartSession() as session:
            e = session.merge(e)
            if e.id is not None:
                session.execute(sa.delete(Exposure).where(Exposure.id == e.id))
                session.commit()

        if fullname is not None and os.path.isfile(fullname):
            os.remove(fullname)
