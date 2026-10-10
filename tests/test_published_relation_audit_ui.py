"""Workbench audit text distinguishes the published subset from full coverage."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_published_relation_cards_do_not_promote_unknowns_or_candidates():
    node=shutil.which("node")
    if node is None:pytest.skip("Node.js is unavailable")
    script=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const start=source.indexOf('function publishedGeometryAuditCards(');
const end=source.indexOf('\nfunction renderIterationAudit',start);
assert(start>=0&&end>start);
const context={};vm.createContext(context);vm.runInContext(source.slice(start,end),context);
const cards=context.publishedGeometryAuditCards;
const strict={certificate_source:'published_validation',current_dxf_verified:true,passed:true,
              satisfied_count:8,required_count:8,angle_tolerance_deg:1e-7};
const incomplete={certificate_source:'published_validation',satisfied:true,all_join_relationships_certified:true,
                  unresolved_joint_count:4,unresolved_arc_arc_joint_count:2,remaining_shape_dof:23,unbound_dimensions:20};
let rows=cards({strict_relation_validation:strict,reconstruction_contract:incomplete});
assert.strictEqual(rows.length,2);assert(rows[0][1].includes('8/8'));
assert(rows[1][1].includes('尚未完整'));assert(rows[1][2].includes('未知接点 4'));
assert(rows[1][2].includes('ARC–ARC 2'));assert(rows[1][2].includes('自由度 23'));
rows=cards({strict_relation_validation:{...strict,certificate_source:'candidate'},
            reconstruction_contract:{...incomplete,unresolved_joint_count:0,remaining_shape_dof:0}});
assert(rows[0][1].includes('尚未通过'));assert(rows[1][1].includes('尚未完整'));
rows=cards({accepted:true,diagnostics:{remaining_shape_dof:0}});
assert(rows[0][1].includes('尚无严格相切证书'));assert(rows[1][1].includes('尚未完整'));
rows=cards({strict_relation_validation:{...strict,required_count:0,satisfied_count:0}});
assert(rows[0][1].includes('未声明相切约束'));assert(rows[1][1].includes('尚未完整'));
rows=cards({strict_relation_validation:strict,reconstruction_contract:{...incomplete,
            unresolved_joint_count:0,unresolved_arc_arc_joint_count:0,remaining_shape_dof:0,unbound_dimensions:0}});
assert(rows[1][1].includes('已识别约束与关系检查通过'));
assert(rows[1][2].includes('不等于 GT 一致'));
"""
    source=Path(__file__).resolve().parents[1]/"web"/"agent.js"
    result=subprocess.run([node,"-e",script,str(source)],capture_output=True,text=True,encoding="utf8",timeout=15)
    assert result.returncode==0,result.stderr
