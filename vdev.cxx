#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include <dlfcn.h>

constexpr int vdev_tag = 0;
constexpr int vdev_rank = 1;
constexpr int host_rank = 0;
MPI_Comm vdev_comm;


enum class Commands : int {
	RMALLOC,
	RFREE,
	MEMCPY_TO_DEV,
	MEMCPY_FROM_DEV,
	KRESOLVE,
	KLAUNCH,
	QUIT
};

struct klaunch_cmd_t {
	void (*kernel)(void*,size_t);
	void* param;
	size_t nb_iter;
};

struct memcpy_cmd_t {
	void* addr;
	size_t size;
};

[[ noreturn ]] 
void vdev_server() {
	for(;;) {
		Commands command;
		MPI_Recv(&command, 1, MPI_INT, host_rank, vdev_tag, vdev_comm, 0);
		switch(command) {
			case Commands::RMALLOC: {
			unsigned long long size;
			MPI_Recv(&size, 1, MPI_UNSIGNED_LONG_LONG, host_rank, vdev_tag, vdev_comm, 0);
			void* malloced = malloc(size);
			unsigned long long result = reinterpret_cast<unsigned long long>(malloced);
			MPI_Send(&result, 1, MPI_UNSIGNED_LONG_LONG, host_rank, vdev_tag, vdev_comm);
		} break;
			case Commands::RFREE: {
			unsigned long long ptr;
			MPI_Recv(&ptr, 1, MPI_UNSIGNED_LONG_LONG, host_rank, vdev_tag, vdev_comm, 0);
			free(reinterpret_cast<void*>(ptr));
		} break;
			case Commands::MEMCPY_TO_DEV: {
			memcpy_cmd_t cpycmd;
			MPI_Recv(&cpycmd, sizeof(cpycmd)/sizeof(char), MPI_CHAR, host_rank, vdev_tag, vdev_comm, 0);
			MPI_Recv(cpycmd.addr, cpycmd.size, MPI_CHAR, host_rank, vdev_tag, vdev_comm, 0);
		} break;
			case Commands::MEMCPY_FROM_DEV: {
			memcpy_cmd_t cpycmd;
			MPI_Recv(&cpycmd, sizeof(cpycmd)/sizeof(char), MPI_CHAR, host_rank, vdev_tag, vdev_comm, 0);
			MPI_Send(cpycmd.addr, cpycmd.size, MPI_CHAR, host_rank, vdev_tag, vdev_comm);
		} break;
			case Commands::KRESOLVE: {
			unsigned long long size;
			MPI_Recv(&size, 1, MPI_UNSIGNED_LONG_LONG, host_rank, vdev_tag, vdev_comm, 0);
			std::string symbol(size, 0);
			MPI_Recv(symbol.data(), size, MPI_CHAR, host_rank, vdev_tag, vdev_comm, 0);
			unsigned long long result = reinterpret_cast<unsigned long long>(dlsym(RTLD_DEFAULT, symbol.c_str()));
			MPI_Send(&result, 1, MPI_UNSIGNED_LONG_LONG, host_rank, vdev_tag, vdev_comm);
		} break;
			case Commands::KLAUNCH: {
			klaunch_cmd_t klnchcmd;
			MPI_Recv(&klnchcmd, sizeof(klaunch_cmd_t)/sizeof(char), MPI_CHAR, host_rank, vdev_tag, vdev_comm, 0);
			klnchcmd.kernel(klnchcmd.param, klnchcmd.nb_iter);
		} break;
			case Commands::QUIT: {
			MPI_Finalize();
			exit(0);
		} break;
		}
	}
}

MPI_Comm vdev_init(MPI_Comm comm){
	int size;
	MPI_Comm_size(comm, &size);
	if ( size % 2 ) {
		std::cerr << " *** Error: even number of MPI ranks required, "<<size<<" found\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	int rank;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_split(comm, rank/2, rank, &vdev_comm);
	int hd_rank;
	MPI_Comm_rank(vdev_comm, &hd_rank);
	MPI_Comm result;
	MPI_Comm_split(comm, hd_rank, rank, &result);
	if ( hd_rank == vdev_rank ) {
		vdev_server();
	} else {
		return result;
	}
}

void vdev_finalize(){
	Commands cmd = Commands::QUIT;
	MPI_Send(&cmd, 1, MPI_INT, vdev_rank, vdev_tag, vdev_comm);
}

void* rmalloc(size_t stsize) {
	Commands cmd = Commands::RMALLOC;
	MPI_Send(&cmd, 1, MPI_INT, vdev_rank, vdev_tag, vdev_comm);
	unsigned long long size=stsize;
	MPI_Send(&size, 1, MPI_UNSIGNED_LONG_LONG, vdev_rank, vdev_tag, vdev_comm);
	unsigned long long result;
	MPI_Recv(&result, 1, MPI_UNSIGNED_LONG_LONG, vdev_rank, vdev_tag, vdev_comm, 0);
	return reinterpret_cast<void*>(result);
}

void rfree(void* vptr) {
	Commands cmd = Commands::RFREE;
	MPI_Send(&cmd, 1, MPI_INT, vdev_rank, vdev_tag, vdev_comm);
	unsigned long long ptr=reinterpret_cast<unsigned long long>(vptr);
	MPI_Send(&ptr, 1, MPI_UNSIGNED_LONG_LONG, vdev_rank, vdev_tag, vdev_comm);
}

void memcpy_to_dev(void* dst, void* src, size_t size) {
	Commands cmd = Commands::MEMCPY_TO_DEV;
	MPI_Send(&cmd, 1, MPI_INT, vdev_rank, vdev_tag, vdev_comm);
	memcpy_cmd_t cpycmd{dst, size};
	MPI_Send(&cpycmd, sizeof(cpycmd)/sizeof(char), MPI_CHAR, vdev_rank, vdev_tag, vdev_comm);
	MPI_Send(src, size, MPI_CHAR, vdev_rank, vdev_tag, vdev_comm);
}

void memcpy_from_dev(void* dst, void* src, size_t size) {
	Commands cmd = Commands::MEMCPY_FROM_DEV;
	MPI_Send(&cmd, 1, MPI_INT, vdev_rank, vdev_tag, vdev_comm);
	memcpy_cmd_t cpycmd{src, size};
	MPI_Send(&cpycmd, sizeof(cpycmd)/sizeof(char), MPI_CHAR, vdev_rank, vdev_tag, vdev_comm);
	MPI_Recv(dst, size, MPI_CHAR, vdev_rank, vdev_tag, vdev_comm, 0);
}

void* kresolve(void* kernel) {
	Commands cmd = Commands::KRESOLVE;
	MPI_Send(&cmd, 1, MPI_INT, vdev_rank, vdev_tag, vdev_comm);
	Dl_info info;
	int retval = dladdr(kernel, &info);
	if ( !info.dli_sname ) {
		std::cerr<<"\n\n *** Error: No symbol name in "<<info.dli_fname<<" for the function at address "<<kernel<<"\n\n\n";
		MPI_Abort(MPI_COMM_WORLD, 2);
	}
	unsigned long long size = strlen(info.dli_sname);
	MPI_Send(&size, 1, MPI_UNSIGNED_LONG_LONG, vdev_rank, vdev_tag, vdev_comm);
	MPI_Send(info.dli_sname, size, MPI_CHAR, vdev_rank, vdev_tag, vdev_comm);
	unsigned long long result;
	MPI_Recv(&result, 1, MPI_UNSIGNED_LONG_LONG, vdev_rank, vdev_tag, vdev_comm, 0);
	return reinterpret_cast<void*>(result);
}

void klaunch(void* kernel, void* param, size_t nb_iter) {
	Commands cmd = Commands::KLAUNCH;
	MPI_Send(&cmd, 1, MPI_INT, vdev_rank, vdev_tag, vdev_comm);
	klaunch_cmd_t klnchcmd{reinterpret_cast<void(*)(void*,size_t)>(kernel), param, nb_iter};
	MPI_Send(&klnchcmd, sizeof(klnchcmd)/sizeof(char), MPI_CHAR, vdev_rank, vdev_tag, vdev_comm);
}

template<class F>
void kwrapper(F* f, size_t nb_iter) {
	for ( size_t it =0; it<nb_iter;++it ) {
		(*f)(it);
	}
}

template<class F>
struct FWrapper {
	F f;
	inline void operator()(size_t it) {
		f(it);
	}
};

template<class F>
void parallel_for(size_t nb_iter, F&& host_f)
{
	void* dev_f = rmalloc(sizeof(host_f));
	memcpy_to_dev(dev_f, &host_f, sizeof(host_f));
	void* kernel = kresolve(reinterpret_cast<void*>(kwrapper<std::remove_cvref_t<decltype(host_f)>>));
	klaunch(kernel, dev_f, nb_iter);
	rfree(dev_f);
}

struct Functor {
	std::array<double, 6>* vect;
	void operator() (size_t ii){
		(*vect)[ii] *=2;
	}
};

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm comm_world = vdev_init(MPI_COMM_WORLD);
	
	std::array<double, 6> host_vect;
	std::cout << "after allocation: ";
	std::ranges::copy(host_vect, std::ostream_iterator<double>(std::cout, ", "));
	std::cout << "\n";
	
	std::ranges::iota(host_vect.begin(), host_vect.end(), 0);
	std::cout << "after initialization: ";
	std::ranges::copy(host_vect, std::ostream_iterator<double>(std::cout, ", "));
	std::cout << "\n";
	
	auto dev_vect = static_cast<decltype(host_vect)*>(rmalloc(sizeof(host_vect)));
	memcpy_to_dev(dev_vect, &host_vect, sizeof(host_vect));
	parallel_for(host_vect.size(), Functor(dev_vect));
	// TODO: I don't manage to get a dynamic symbol when lambdas are involved:
	// parallel_for(host_vect.size(), [=](size_t ii){
	// 	(*dev_vect)[ii] *= 2;
	// });
	std::cout << "after computation: ";
	std::ranges::copy(host_vect, std::ostream_iterator<double>(std::cout, ", "));
	std::cout << "\n";
	
	memcpy_from_dev(&host_vect, dev_vect, sizeof(host_vect));
	std::cout << "after copy back: ";
	std::ranges::copy(host_vect, std::ostream_iterator<double>(std::cout, ", "));
	std::cout << "\n";
	
	rfree(dev_vect);
	
	vdev_finalize();
	MPI_Finalize();
}
