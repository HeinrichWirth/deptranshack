"""Mechanical orchestration adapter; original geometry/selection AST retained."""
import ast,inspect,textwrap
import long_tracker as frozen

def function(tree,namespace,name):
    ast.fix_missing_locations(tree);exec(compile(tree,'<C4 batch orchestration>','exec'),namespace)
    return namespace[name]

def make_infer(tracker_type):
    # Turn only fit calls into yield points, leaving all pre/post math unchanged.
    class YieldFit(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='tt' and node.func.attr=='fit':
                assert not node.keywords
                return ast.Yield(ast.Tuple(node.args,ast.Load()))
            return node
    gen=function(YieldFit().visit(ast.parse(textwrap.dedent(inspect.getsource(frozen.candidate_records)))),dict(frozen.candidate_records.__globals__),'candidate_records')
    def prepare(active,source,tt,template,cfg,k):
        rows=[];waiting=[]
        for p in active:
            try:frame=frozen.frame_for(p,cfg)
            except ValueError as e:rows.append(dict(error=str(e)));continue
            C,B,old,curv,kappa,ang=frame;g=gen(source,dict(p,angles=ang),tt,template,cfg,C,B,old,curv,kappa,k)
            row=dict(frame=frame,generator=g);rows.append(row)
            try:pack=next(g);waiting.append((row,pack))
            except StopIteration as e:row['result']=e.value
        while waiting:
            values=tt.fit_batch([p for row,p in waiting]);following=[]
            for (row,pack),value in zip(waiting,values):
                try:following.append((row,row['generator'].send(value)))
                except StopIteration as e:row['result']=e.value
            waiting=following
        return rows
    def frame(rows,i):
        if 'error' in rows[i]:raise ValueError(rows[i]['error'])
        return rows[i]['frame']
    class Batch(ast.NodeTransformer):
        def visit_For(self,node):
            node=self.generic_visit(node)
            if isinstance(node.target,ast.Tuple) and [x.id for x in node.target.elts if isinstance(x,ast.Name)]==['parent_id','p']:
                setup=ast.parse('prepared_batch = prepare_batch(active,source,tt,template,cfg,k)').body[0]
                return [setup,node]
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='frame_for':return ast.parse('prepared_frame(prepared_batch,parent_id)',mode='eval').body
            if isinstance(node.func,ast.Name) and node.func.id=='candidate_records':return ast.parse("prepared_batch[parent_id]['result']",mode='eval').body
            return node
    ns=dict(frozen.infer.__globals__,TemplateTracker=tracker_type,prepare_batch=prepare,prepared_frame=frame)
    return function(Batch().visit(ast.parse(textwrap.dedent(inspect.getsource(frozen.infer)))),ns,'infer')
