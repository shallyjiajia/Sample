class ClassUtil():
    def newInstance(kls):
        parts = kls.split('.')
        module = ".".join(parts[:-1])
        print(module)
        m = __import__( module )
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m()

if __name__=='__main__':
    testbid = ClassUtil.newInstance('generatorbid.bidservice.BidService')
    testbid.toString()