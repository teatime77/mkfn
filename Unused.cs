using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace MkFn {
    public partial class MkFn {

        /*
            foreach以外の文を削除する。
        */
        void ForEachSkeleton(Function fnc) {
            Traverse(fnc,
                delegate (object obj) {
                    if (obj is BlockStatement) {
                        // ブロック文の場合

                        BlockStatement block = obj as BlockStatement;

                        // foreach以外の文を削除する。
                        block.Statements = (from x in block.Statements where x is ForEach select x).ToList();
                    }
                });
        }

        /*
            ループ変数に対応するfor文を探し、なければfor文を作る。
        */
        ForEach FindForEach(BlockStatement blc, List<Variable> loop_vars, Dictionary<Variable, Variable> loop_var_tbl) {
            // 現在にブロック文に含まれるfor文のリスト
            List<ForEach> for_list = (from x in blc.Statements where x is ForEach select x as ForEach).ToList();

            // 未処理のループ変数に対し
            foreach (Variable va in loop_vars) {
                // ループ変数の定義域と同じfor文のリスト
                var v = from f in for_list where f.LoopVariable.Domain.Eq(va.Domain) select f;
                if (v.Any()) {
                    // ループ変数の定義域と同じfor文がある場合

                    // ループ変数の定義域と同じfor文
                    ForEach for1 = v.First();

                    // 未処理のループ変数のリストから取り除く。
                    loop_vars.Remove(va);

                    // 辞書( ループ変数→for文のループ変数 ) に追加する。
                    loop_var_tbl.Add(va, for1.LoopVariable);

                    if (loop_vars.Count == 0) {
                        // 未処理のループ変数がない場合

                        // 現在のfor文を返す。
                        return for1;
                    }
                    else {
                        // 未処理のループ変数がある場合

                        // 再帰呼び出しをする。
                        return FindForEach(for1, loop_vars, loop_var_tbl);
                    }
                }
            }

            // 現在のブロック文
            BlockStatement current_blc = blc;

            // 未処理のループ変数がある間
            while (loop_vars.Any()) {

                Variable va = loop_vars[0];

                // 未処理のループ変数のリストから取り除く。
                loop_vars.Remove(va);

                // ループ変数をコピーする。
                Variable va2 = va.Clone();

                // for文を作る。
                ForEach for1 = new ForEach(va2, new List<Statement>());

                // 辞書( ループ変数→for文のループ変数 ) に追加する。
                loop_var_tbl.Add(va, for1.LoopVariable);

                // 現在のブロック文にfor文を追加する。
                current_blc.AddStatement(for1);

                // for文を現在のブロック文にする。
                current_blc = for1;
            }

            // 現在のブロック文(for文)を返す。
            return current_blc as ForEach;
        }

        void SetupForEach(Function forward, List<Assignment> sorted_backward_asns) {
            // 順伝播の関数をコピーする。
            Function backward_fnc = forward.Clone() as Function;

            // 関数名をBackwardに変える。
            backward_fnc.Name = "Backward";

            // foreach以外の文を削除する。
            ForEachSkeleton(backward_fnc);

            // for文のリスト
            List<ForEach> for_list = new List<ForEach>();

            // 逆伝播のすべての代入文に対し
            foreach (Assignment asn in sorted_backward_asns) {

                // 代入文の中で参照されているループ変数のリスト
                List<Variable> loop_vars = (from r in EnumReference(asn) where r.VarRef.ParentVar == FreeVariable || r.VarRef.ParentVar is ForEach select r.VarRef).Distinct().ToList();
                if (loop_vars.Any()) {
                    // ループ変数がある場合

                    Dictionary<Variable, Variable> loop_var_tbl = new Dictionary<Variable, Variable>();

                    // ループ変数に対応するfor文を探し、なければfor文を作る。
                    ForEach for1 = FindForEach(backward_fnc.BodyStatement, loop_vars, loop_var_tbl);

                    // for文に代入文を追加する。
                    for1.AddStatement(asn.Clone(loop_var_tbl));

                    if (!for_list.Contains(for1)) {
                        // for文のリストに含まれない場合

                        for_list.Add(for1);
                    }
                }
            }
        }

        /*
            共通部分式除去

            スレッドとブロックのIDから添え字を計算。
            同じ添え字の計算式をまとめる。        
        */
        void CommonSubexpressionElimination(BlockStatement blc) {
            // ブロック文の中の多次元の添え字付きの変数参照のリスト
            List<Reference> refs = AllRefs(blc).Where(r => r.Indexes != null && r.Indexes.Length != 1).ToList();

            Dictionary<Term, Variable> common_offsets = new Dictionary<Term, Variable>(new TermEqualityComparer());

            foreach (Reference rf in refs) {
                Term offset = OffsetFromIndexes(rf);

                Variable va;

                if (common_offsets.TryGetValue(offset, out va)) {

                }
                else {

                    va = new Variable(offset.ToString(), IntClass, null);
                    common_offsets.Add(offset, va);
                }
            }
        }
    }

    public class TermEqualityComparer : IEqualityComparer<Term> {
        bool IEqualityComparer<Term>.Equals(Term x, Term y) {
            return x.Eq(y);
        }

        int IEqualityComparer<Term>.GetHashCode(Term obj) {
            return obj.HashCode();
        }
    }
}