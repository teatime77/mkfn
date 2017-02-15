using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MkFn {
    public partial class MkFn {

        /*
            LINQの微分
        */
        Term DifferentialLINQ(LINQ lnq, Reference r1, Dictionary<Variable, Variable> var_tbl) {
            Debug.Assert(lnq.Aggregate != null);

            Dictionary<Reference, Dictionary<Reference, Term>> rs = new Dictionary<Reference, Dictionary<Reference, Term>>();
            bool exact = false;
            Traverse(lnq.Select,
                delegate (object obj) {
                    if (obj is Reference) {
                        // 変数参照の場合

                        Reference r2 = obj as Reference;

                        if (!(from r in rs.Keys where r.Eq(r2) select r).Any()) {
                            // 処理済みでない場合

                            if (r1.VarRef == r2.VarRef) {
                                // 同じ変数を参照する場合

                                if (r1.Eq(obj)) {
                                    // 一致する場合

                                    exact = true;
                                }
                                else {
                                    // 一致しない添え字がある場合

                                    Dictionary<Reference, Term> pairs = new Dictionary<Reference, Term>();
                                    bool ok = true;
                                    for (int i = 0; i < r1.Indexes.Length; i++) {
                                        if (!r1.Indexes[i].Eq(r2.Indexes[i])) {
                                            // 添え字が一致しない場合

                                            if (!(r2.Indexes[i] is Reference)) {
                                                // 代入候補の変数参照の添え字が変数参照でない場合

                                                ok = false;
                                                break;
                                            }
                                            else {
                                                // 両方の添え字が変数参照の場合

                                                Reference r3 = r2.Indexes[i] as Reference;
                                                IEnumerable<Variable> linq_eq_vars = from v in lnq.Variables where v == r3.VarRef select v;
                                                if (linq_eq_vars.Any()) {
                                                    // LINQの変数の場合

                                                    Variable v = linq_eq_vars.First();
                                                    Debug.Assert(!(from r in pairs.Keys where r.VarRef == v select r).Any());
                                                    pairs.Add(new Reference(v), r1.Indexes[i]);
                                                }
                                                else {
                                                    // LINQの変数でない場合

                                                    ok = false;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    if (ok) {
                                        rs.Add(r2, pairs);
                                    }
                                }
                            }
                        }
                    }
                });

            Debug.Assert(!(exact && rs.Any()), "完全一致の変数参照と代入で一致の変数参照の両方がある場合は未対応");
            if (!(exact || rs.Any())) {
                // 完全一致や代入で一致の変数参照がない場合

                return Zero();
            }

            // LINQをコピーする。
            LINQ lnq1 = lnq.Clone(var_tbl);

            if (exact) {
                // 完全一致の変数参照がある場合

                // select句を微分する。
                Term dif1 = Differential(lnq1.Select, r1, var_tbl);

                if (lnq.Aggregate.Name == "Sum") {
                    return lnq1;
                }
                else if (lnq.Aggregate.Name == "Prod") {
                    Debug.Write("の微分は未実装");
                    return lnq1;
                }
                if (lnq.Aggregate.Name == "Max") {
                    Debug.Write("の微分は未実装");
                    return lnq1;
                }
                else {
                    Debug.Assert(false);
                }
            }
            else {
                // 代入で一致の変数参照がある場合

                Debug.Assert(rs.Keys.Count == 1, "代入で一致の変数参照は1種類のみ実装");
                Dictionary<Reference, Term> subst_tbl = rs.First().Value;
                Debug.Assert(subst_tbl.Count == lnq.Variables.Length, "LINQの全変数に代入する。");

                // LINQのselect句の変数参照に代入する。
                Term subst_sel = Subst(lnq1.Select, subst_tbl, var_tbl);

                // LINQの変数に代入をしたselect句を微分する。
                Term dif1 = Differential(subst_sel, r1, var_tbl);

                if (lnq.Aggregate.Name == "Sum") {
                    return dif1;
                }
                else if (lnq.Aggregate.Name == "Prod") {
                    Debug.Write("の微分は未実装");
                    return dif1;
                }
                if (lnq.Aggregate.Name == "Max") {
                    Debug.Write("の微分は未実装");
                    return lnq1;
                }
                else {
                    Debug.Assert(false);
                }
            }

            return null;
        }

        /*
            微分
        */
        Term Differential(Term t1, Reference r1, Dictionary<Variable, Variable> var_tbl_up) {
            if (t1 is Reference) {
                // 変数参照の場合

                if (t1.Eq(r1)) {
                    return One();
                }
                else {
                    return Zero();
                }
            }
            else if (t1 is Number) {
                // 数値定数の場合

                return Zero();
            }

            Dictionary<Variable, Variable> var_tbl = (var_tbl_up == null ? new Dictionary<Variable, Variable>() : new Dictionary<Variable, Variable>(var_tbl_up));
            if (t1 is Apply) {
                // 関数適用の場合

                Apply app = t1 as Apply;

                Term[] diffs = (from t in app.Args select Differential(t, r1, var_tbl)).ToArray();

                if (app.Function.VarRef == AddFnc) {
                    // 加算の場合

                    return Add(diffs);
                }
                else if (app.Function.VarRef == MulFnc) {
                    // 乗算の場合

                    Term[] args = new Term[app.Args.Length];
                    foreach (int i in Range(app.Args.Length)) {
                        args[i] = Mul((from j in Range(app.Args.Length) select (i == j ? diffs[i] : app.Args[j].Clone(var_tbl))).ToArray());
                    }

                    return Add(args);
                }
                else if (app.Function.VarRef == MaxPoolFnc) {
                    return new Apply(MaxPoolPrimeFnc, app.Args[0].Clone(var_tbl));
                }
                else if (app.Function.Name == "σ") {

                    Term[] args = (from t in app.Args select t.Clone(var_tbl)).ToArray();
                    return Mul(new Apply(new Reference(σ_prime), args), diffs[0]);
                }
                else if (app.Function.Name == "tanh") {

                    Term[] args = (from t in app.Args select t.Clone(var_tbl)).ToArray();
                    return Mul(new Apply(new Reference(tanh_prime), args), diffs[0]);
                }
                else if (app.Function.Name == "Mat" || app.Function.Name == "Row" || app.Function.Name == "C") {
                    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 未実装 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    return Zero();
                }
                else {
                    Debug.Assert(false);
                }
            }
            else if (t1 is LINQ) {
                // LINQの場合

                return DifferentialLINQ(t1 as LINQ, r1, var_tbl);
            }
            else {
                Debug.Assert(false);
            }

            Debug.Assert(false);
            return null;
        }

        /*
            変数に項を代入する。
        */
        Term Subst(Term t1, Dictionary<Reference, Term> subst_tbl, Dictionary<Variable, Variable> var_tbl = null) {
            return TraverseRep(t1,
                delegate (object obj, out object ret) {
                    ret = obj;
                    if (obj is Reference) {
                        // 変数参照の場合

                        var vref = from r in subst_tbl.Keys where r.Eq(obj) select r;
                        if (vref.Any()) {

                            ret = subst_tbl[vref.First()].Clone(var_tbl);
                            return true;
                        }
                    }

                    return false;
                }) as Term;
        }

        /*
            数式の簡約化
        */
        Term SimplifyExpression(Term t1) {
            return TraverseRep(t1,
                delegate (object obj, out object ret) {
                    ret = obj;

                    if (obj is Apply) {
                        // 関数適用の場合

                        Apply app = obj as Apply;

                        // 引数を簡約化する。
                        Term[] args1 = (from t in app.Args select SimplifyExpression(t)).ToArray();


                        if (app.IsAdd() || app.IsMul()) {

                            List<Term> args2 = new List<Term>();

                            foreach (Term t2 in args1) {
                                if (t2 is Apply && (t2 as Apply).Function.VarRef == app.Function.VarRef) {
                                    // 引数が同じ演算の場合

                                    args2.AddRange(t2.AsApply().Args);
                                }
                                else {
                                    // 引数が加算や減算でない場合

                                    args2.Add(t2);
                                }
                            }

                            for (int i = 0; i < args2.Count; i++) {

                                for (int j = i + 1; j < args2.Count;) {
                                    if (args2[i].EqBody(args2[j])) {
                                        // 本体が同じ項がある場合

                                        if (app.IsAdd()) {
                                            // 加算の場合

                                            // 係数を加算する。
                                            args2[i].Value += args2[j].Value;
                                        }
                                        else {
                                            // 乗算の場合

                                            // 係数を乗算する。
                                            args2[i].Value *= args2[j].Value;
                                        }

                                        // 同じ項を取り除く
                                        args2.RemoveAt(j);
                                    }
                                    else {
                                        // 同じ項がない場合

                                        j++;
                                    }
                                }
                            }

                            if (app.IsAdd()) {
                                // 加算の場合

                                // 係数が0の項を除く。
                                Term[] args3 = (from t in args2 where t.Value != 0 select t).ToArray();

                                switch (args3.Length) {
                                case 0:
                                    ret = Zero();
                                    return true;

                                case 1:
                                    ret = args3[0];

                                    return true;

                                default:
                                    ret = Add(args3);
                                    return true;
                                }
                            }
                            else {
                                // 乗算の場合

                                // 引数の係数をすべてかけてまとめる。
                                double n = (from t in args2 select t.Value).Aggregate((x, y) => x * y);
                                foreach (Term t in args2) {
                                    t.Value = 1;
                                }

                                if (n == 0) {
                                    // 係数の積が0の場合

                                    // 結果は0
                                    ret = Zero();
                                    return true;
                                }

                                // 定数を除く。
                                Term[] args3 = (from t in args2 where !(t is Number) select t).ToArray();

                                switch (args3.Length) {
                                case 0:
                                    Debug.Assert(args2.Count == 1 && args2[0] is Number && (args2[0] as Number).Value == 1);

                                    ret = One();
                                    break;

                                case 1:
                                    ret = args3[0];
                                    break;

                                default:
                                    ret = Mul(args3);
                                    break;
                                }

                                (ret as Term).Value = app.Value * n;
                                return true;
                            }
                        }

                        ret = new Apply(app.Function.VarRef, args1);

                        return true;
                    }

                    return false;
                }) as Term;
        }
    }
}