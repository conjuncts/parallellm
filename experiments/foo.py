from parallellm.core.gateway import Parallellm
# from parallellm.core.manager import BatchManager


print("Before")

# mgr = BatchManager()
mgr = Parallellm.resume_directory(".enzy")

with mgr:
    mgr.when_stage("init")
    print("Inside init stage")

    mgr.goto_stage("next")

with mgr:
    mgr.when_stage("next")
    print("Inside next stage")
print("After")

mgr.persist()
