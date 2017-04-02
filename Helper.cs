using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MkFn {
    public partial class MkFn {
        /*
            newならtrueを返します。
        */
        public static bool IsNew(Term t) {
            return t is Apply && t.AsApply().Function.VarRef == MkFn.Singleton.NewFnc;
        }

        /*
            Rangeならtrueを返します。
        */
        public static bool IsRange(Term t) {
            return t is Apply && t.AsApply().Function.VarRef == MkFn.Singleton.RangeFnc;
        }

        /*
            tならtrueを返します。
        */
        public static bool Is_t(object obj) {
            if(obj is Variable) {

                return (obj as Variable).Name == "t";
            }
            else if (obj is Reference) {

                return (obj as Reference).Name == "t";
            }

            return false;
        }

        /*
            Tならtrueを返します。
        */
        public static bool Is_T(object obj) {
            if (obj is Variable) {

                return (obj as Variable).Name == "T";
            }
            else if (obj is Reference) {

                return (obj as Reference).Name == "T";
            }

            return false;
        }

        /*
            t+1ならtrueを返します。
        */
        public static bool Is_t_plus(Term t) {
            if( t is Apply && t.AsApply().Function.VarRef == MkFn.Singleton.AddFnc) {
                Apply app = t as Apply;

                if(app.Args.Length == 2 && app.Args[0] is Reference && (app.Args[0] as Reference).Name == MkFn.Singleton.t_var.Name) {
                    if (app.Args[1] is Number && app.Args[1].Value == 1) {

                        return true;
                    }
                }
            }

            return false;
        }

        /*
            t-1ならtrueを返します。
        */
        public static bool Is_t_minus(Term t) {
            if (t is Apply && t.AsApply().Function.VarRef == MkFn.Singleton.AddFnc) {
                Apply app = t as Apply;

                if (app.Args.Length == 2 && app.Args[0] is Reference && (app.Args[0] as Reference).Name == MkFn.Singleton.t_var.Name) {
                    if (app.Args[1] is Number && app.Args[1].Value == -1) {

                        return true;
                    }
                }
            }

            return false;
        }

        Apply Range(Term start, Term end) {
            return new Apply(RangeFnc, start, end);
        }

        Apply Intersect(Term t1, Term t2) {
            if (t1 is Apply && t2 is Apply) {
                // 関数適用の場合

                Apply app1 = t1 as Apply;
                Apply app2 = t2 as Apply;

                if (app1.Function.VarRef == RangeFnc && app2.Function.VarRef == RangeFnc) {

                    Apply min = new Apply(maxFnc, new Term[] { MinRange(t1), MinRange(t2) });
                    Apply max = new Apply(minFnc, new Term[] { MaxRange(t1), MaxRange(t2) });

                    return Range(min, max);
                }
            }
            return null;
        }

        Term MinRange(Term rng) {
            if (rng is Apply) {
                // 関数適用の場合

                Apply app = rng as Apply;

                if (app.Function.VarRef == RangeFnc) {

                    if (app.Args.Length == 1) {

                        return Zero();
                    }
                    else {

                        return app.Args[0];
                    }
                }
            }

            throw new Exception();
        }

        Term MaxRange(Term rng) {
            if (rng is Apply) {
                // 関数適用の場合

                Apply app = rng as Apply;

                if (app.Function.VarRef == RangeFnc) {

                    if (app.Args.Length == 1) {

                        return app.Args[0];
                    }
                    else {

                        return app.Args[1];
                    }
                }
            }

            throw new Exception();
        }

        Apply Diff(Term t, Reference r) {
            return new Apply(DiffFnc, new Term[] { t, r });
        }

        static Term[] VariableToReference(object[] args) {
            return (from x in args select (Term)(x is Variable ? new Reference(x as Variable) : x as Term)).ToArray();
        }

        static Apply Mul(params object[] args) {
            return new Apply(new Reference(Singleton.MulFnc), VariableToReference(args));
        }

        static Apply Div(params object[] args) {
            return new Apply(new Reference(Singleton.DivFnc), VariableToReference(args));
        }

        static Apply Add(params object[] args) {
            return new Apply(new Reference(Singleton.AddFnc), VariableToReference(args));
        }

        Apply Sub(params object[] args) {
            Term[] v = VariableToReference(args);
            for (int i = 1; i < v.Length; i++) {
                v[i].Value *= -1;
            }

            return new Apply(new Reference(AddFnc), v);
        }

        public static int[] Range(int n) {
            int[] v = new int[n];

            for (int i = 0; i < n; i++) {
                v[i] = i;
            }

            return v;
        }

        Term SetParent(Term t1) {
            return TraverseRep(t1,
                delegate (object obj, out object ret) {
                    ret = obj;

                    return false;
                }) as Term;

        }

        /*
            内部の変数参照のリストを返します。
        */
        Reference[] EnumReference(object root) {
            List<Reference> v = new List<Reference>();
            Traverse(root,
                delegate (object obj) {
                    if (obj is Reference) {
                        // 変数参照の場合

                        v.Add(obj as Reference);
                    }
                });
            return v.ToArray();
        }

        /*
            項を含む文を返します。
        */
        Statement ParentStatement(Term t1) {
            for (Object obj = t1.Parent; ;) {
                Debug.Assert(obj != null);

                if (obj is Statement) {
                    // 文の場合

                    return obj as Statement;
                }
                else if (obj is Term) {
                    // 項の場合

                    obj = (obj as Term).Parent;
                }
                else if (obj is Variable) {
                    // 変数の場合

                    obj = (obj as Variable).ParentVar;
                }
                else {
                    Debug.Assert(false);
                }
            }
        }

        /*
            文の祖先のForEachのリストを返します。
        */
        ForEach[] AncestorForEach(Statement stmt) {
            List<ForEach> vfor = new List<ForEach>();
            Debug.Assert(stmt.ParentStmt is ForEach);

            ForEach for1 = stmt.ParentStmt as ForEach;
            vfor.Add(for1);
            while (for1.ParentStmt is ForEach) {
                for1 = for1.ParentStmt as ForEach;
                vfor.Add(for1);
            }
            Debug.Assert(for1.ParentStmt is BlockStatement);

            // 1番外側のForEachがリストの最初になるように順序を反転させます。
            vfor.Reverse();

            return vfor.ToArray();
        }

        /*
            木構造を走査します。
        */
        public static void Traverse(object obj, NaviAction before, NaviAction after = null) {
            if (obj == null) {
                return;
            }

            if (before != null) {
                before(obj);
            }

            if (obj is Term) {
                // 項の場合

                if (obj is Reference) {
                    // 変数参照の場合

                    Reference r1 = obj as Reference;

                    if (r1.Indexes != null) {
                        foreach (Term t in r1.Indexes) {
                            Traverse(t, before, after);
                        }
                    }
                }
                else if (obj is Number) {
                    // 数値定数の場合

                }
                else if (obj is Apply) {
                    // 関数適用の場合

                    Apply app = obj as Apply;
                    Traverse(app.Function, before, after);
                    foreach (Term t in app.Args) {
                        Traverse(t, before, after);
                    }
                }
                else if (obj is LINQ) {
                    // LINQの場合

                    LINQ lnq = obj as LINQ;
                    foreach (Variable v in lnq.Variables) {
                        Traverse(v, before, after);
                    }
                    Traverse(lnq.Select, before, after);
                    Traverse(lnq.Aggregate, before, after);
                }
                else {
                    Debug.Assert(false);
                }
            }
            else if (obj is Variable) {
                // 変数の場合

                Variable v = obj as Variable;

                Traverse(v.Domain, before, after);

                if (obj is Function) {
                    // 関数の場合

                    Function fnc = obj as Function;
                    foreach (Variable p in fnc.Params) {
                        Traverse(p, before, after);
                    }
                    Traverse(fnc.BodyStatement, before, after);
                }
            }
            else if (obj is Statement) {
                // 文の場合

                if (obj is Assignment) {
                    Assignment asn = obj as Assignment;
                    Traverse(asn.Left, before, after);
                    Traverse(asn.Right, before, after);
                }
                else if (obj is Return) {
                    // returnの場合

                    Return ret = obj as Return;
                    Traverse(ret.Value, before, after);
                }
                else if (obj is ForEach) {
                    // foreachの場合

                    ForEach for1 = obj as ForEach;
                    foreach(Variable va in for1.LoopVariables) {
                        Traverse(va, before, after);
                    }
                    foreach (Statement s in for1.Statements) {
                        Traverse(s, before, after);
                    }
                }
                else if (obj is BlockStatement) {
                    // ブロック文の場合

                    BlockStatement blc1 = obj as BlockStatement;
                    foreach (Statement s in blc1.Statements) {
                        Traverse(s, before, after);
                    }
                }
                else {
                    Debug.Assert(false);
                }
            }
            else if (obj is Class) {
                // クラスの場合

                Class cls = obj as Class;
                foreach (Variable fld in cls.Fields) {
                    Traverse(fld, before, after);
                }
                foreach (Function fnc in cls.Functions) {
                    Traverse(fnc, before, after);
                }
            }
            else {
                Debug.Assert(false);
            }

            if (after != null) {

                after(obj);
            }
        }

        /*
            木構造を走査してノードを置換します。
        */
        object TraverseRep(object obj, NaviFnc before, NaviFnc after = null) {
            if (obj == null) {
                return null;
            }

            object ret = obj;

            if (before != null) {

                bool done = before(obj, out ret);
                if (done) {

                    return ret;
                }
            }

            if (obj is Term) {
                // 項の場合

                if (obj is Reference) {
                    // 変数参照の場合

                    Reference r1 = obj as Reference;

                    if (r1.Indexes != null) {
                        r1.Indexes = (from t in r1.Indexes select TraverseRep(t, before, after) as Term).ToArray();

                        foreach (Term t in r1.Indexes) {
                            t.Parent = obj;
                        }
                    }
                }
                else if (obj is Number) {
                    // 数値定数の場合

                }
                else if (obj is Apply) {
                    // 関数適用の場合

                    Apply app = obj as Apply;
                    app.Function = TraverseRep(app.Function, before, after) as Reference;
                    app.Args = (from t in app.Args select TraverseRep(t, before, after) as Term).ToArray();

                    app.Function.Parent = app;
                    foreach (Term t in app.Args) {
                        t.Parent = obj;
                    }
                }
                else if (obj is LINQ) {
                    // LINQの場合

                    LINQ lnq = obj as LINQ;
                    lnq.Variables = (from v in lnq.Variables select TraverseRep(v, before, after) as Variable).ToArray();
                    lnq.Select = TraverseRep(lnq.Select, before, after) as Term;
                    lnq.Aggregate = TraverseRep(lnq.Aggregate, before, after) as Reference;

                    foreach (Variable v in lnq.Variables) {
                        v.ParentVar = obj;
                    }
                    lnq.Select.Parent = obj;
                    if (lnq.Aggregate != null) {
                        lnq.Aggregate.Parent = obj;
                    }
                }
                else {
                    Debug.Assert(false);
                }
            }
            else if (obj is Variable) {
                // 変数の場合

                Variable v = obj as Variable;

                v.Domain = TraverseRep(v.Domain, before, after) as Term;
                if (v.Domain != null) {
                    v.Domain.Parent = obj;
                }

                if (obj is Function) {
                    // 関数の場合

                    Function fnc = obj as Function;
                    fnc.Params = (from p in fnc.Params select TraverseRep(p, before, after) as Variable).ToList();
                    fnc.BodyStatement = TraverseRep(fnc.BodyStatement, before, after) as BlockStatement;

                    foreach (Variable p in fnc.Params) {
                        p.ParentVar = fnc;
                    }
                    fnc.BodyStatement.ParentStmt = fnc;
                }
            }
            else if (obj is Statement) {
                // 文の場合

                if (obj is Assignment) {
                    Assignment asn = obj as Assignment;
                    asn.Left = TraverseRep(asn.Left, before, after) as Reference;
                    asn.Right = TraverseRep(asn.Right, before, after) as Term;

                    asn.Left.Parent = obj;
                    asn.Right.Parent = obj;
                }
                else if (obj is Return) {
                    // returnの場合

                    Return ret_stmt = obj as Return;
                    ret_stmt.Value = TraverseRep(ret_stmt.Value, before, after) as Term;

                    if (ret_stmt.Value != null) {

                        ret_stmt.Value.Parent = obj;
                    }
                }
                else if (obj is ForEach) {
                    // foreachの場合

                    ForEach for1 = obj as ForEach;
                    for1.LoopVariables = (from va in for1.LoopVariables select TraverseRep(va, before, after) as Variable).ToList();
                    for1.Statements = (from s in for1.Statements select TraverseRep(s, before, after) as Statement).ToList();

                    foreach(Variable va in for1.LoopVariables) {
                        va.ParentVar = obj;
                    }
                    foreach (Statement stmt in for1.Statements) {
                        stmt.ParentStmt = obj;
                    }
                }
                else if (obj is BlockStatement) {
                    // ブロック文の場合

                    BlockStatement blc1 = obj as BlockStatement;
                    blc1.Statements = (from s in blc1.Statements select TraverseRep(s, before, after) as Statement).ToList();
                    foreach (Statement stmt in blc1.Statements) {
                        stmt.ParentStmt = obj;
                    }
                }
                else {
                    Debug.Assert(false);
                }
            }
            else if (obj is Class) {
                // クラスの場合

                Class cls = obj as Class;
                cls.Fields = (from fld in cls.Fields select TraverseRep(fld, before, after) as Variable).ToList();
                cls.Functions = (from fnc in cls.Functions select TraverseRep(fnc, before, after) as Function).ToList();

                foreach (Variable fld in cls.Fields) {
                    fld.ParentVar = cls;
                }
                foreach (Function fnc in cls.Functions) {
                    fnc.ParentVar = cls;
                }
            }
            else {
                Debug.Assert(false);
            }

            if (after != null) {

                after(obj, out ret);
            }

            return ret;
        }

        /*
            木構造に含まれる変数参照のリストを返します。
        */
        public static List<Reference> AllRefs(object root) {
            List<Reference> all_refs = new List<Reference>();

            MkFn.Traverse(root,
                delegate (object obj) {
                    if (obj is Reference) {
                        // 変数参照の場合

                        all_refs.Add(obj as Reference);
                    }
                });

            return all_refs;
        }

        /*
            木構造に含まれる変数参照のリストを返します。
        */
        public static List<object> All(object root, Type tp) {
            List<object> alls = new List<object>();

            MkFn.Traverse(root,
                delegate (object obj) {
                    if (tp.IsInstanceOfType(obj)) {
                        // 変数参照の場合

                        alls.Add(obj);
                    }
                });

            return alls;
        }

        public static List<T> All<T>(object root) {
            List<T> alls = new List<T>();

            MkFn.Traverse(root,
                delegate (object obj) {
                    if (obj is T) {
                        // 変数参照の場合

                        alls.Add((T)obj);
                    }
                });

            return alls;
        }

        static bool EqList(IEnumerable e1, IEnumerable e2) {
            List<object> v1 = new List<object>();
            IEnumerator enum1 = e1.GetEnumerator();
            while (enum1.MoveNext()) {
                v1.Add(enum1.Current);
            }

            List<object> v2 = new List<object>();
            IEnumerator enum2 = e2.GetEnumerator();
            while (enum2.MoveNext()) {
                v2.Add(enum2.Current);
            }

            if(v1.Count != v2.Count) {
                return false;
            }

            while(v1.Count != 0) {
                object o1 = v1[0];
                v1.RemoveAt(0);

                int idx = v2.IndexOf(o1);
                if(idx == -1) {
                    return false;
                }
                v2.RemoveAt(idx);
            }

            return true;
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
