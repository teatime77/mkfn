using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace MkFn {
    public partial class MkFn {
        /*
         順伝播の関数内の代入文の表明 
        */
        void AssertForwardAssignment(List<Assignment> forward_asns) {
            // すべての代入文に対し
            foreach (Assignment asn in forward_asns) {

                // 代入文の左辺の変数参照
                Reference left = asn.Left as Reference;
                Debug.Assert(left.Indexes != null);

                // 左辺の変数参照の次元
                int dim_cnt = left.Indexes.Length;

                // 代入文の祖先のForEachのリスト
                List<Variable> loop_vars = (from x in AncestorForEach(asn) from va in x.LoopVariables select va).ToList();
                Debug.Assert(loop_vars.Count == dim_cnt);

                // 左辺の変数参照の各添え字に対し
                for (int dim = 0; dim < dim_cnt; dim++) {
                    if (left.Name == "a" && dim == 1) {

                        Debug.WriteLine("a[t, φ[t, n]] = (1 - u[t, φ[t, n]]) * Prod(from i in Range(n) select u[t, φ[t, i]]);");
                    }
                    else {

                        // 左辺の変数参照の添え字 = 代入文の祖先のForEachの変数
                        Debug.Assert(left.Indexes[dim] is Reference && (left.Indexes[dim] as Reference).VarRef == loop_vars[dim]);
                    }
                }
            }
        }

        /*
            VarRefとTypeVarの表明
        */
        void AssertVarRefTypeVar(Class cls) {
            Traverse(cls,
                delegate (object obj) {
                    if (obj is Reference) {
                            // 変数参照の場合

                            Debug.Assert((obj as Reference).VarRef != null);
                    }
                    else if (obj is Variable) {
                            // 変数の場合

                            Debug.Assert((obj as Variable).TypeVar != null);
                    }
                });
        }
    }
}